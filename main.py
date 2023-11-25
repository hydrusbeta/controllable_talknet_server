import argparse
import base64
import json
import os
import os.path
import subprocess
import traceback

import hay_say_common as hsc
import jsonschema
import soundfile
from flask import Flask, request
from hay_say_common.cache import Stage
from jsonschema.exceptions import ValidationError

ARCHITECTURE_NAME = 'controllable_talknet'
ARCHITECTURE_ROOT = os.path.join(hsc.ROOT_DIR, ARCHITECTURE_NAME)
RESULTS_DIR = os.path.join(ARCHITECTURE_ROOT, 'results')
PYTHON_EXECUTABLE = os.path.join(hsc.ROOT_DIR, '.venvs', 'controllable_talknet', 'bin', 'python')
INFERENCE_CODE_PATH = os.path.join(ARCHITECTURE_ROOT, 'command_line_interface.py')

TALKNET_INPUT_FORMAT, TALKNET_INPUT_EXTENSION = 'WAV', '.wav'
REFERENCE_PREFIX = 'Reference_'

app = Flask(__name__)


def register_methods(cache):
    @app.route('/generate', methods=['POST'])
    def generate() -> (str, int):
        code = 200
        message = ""
        try:
            user_text, input_filename_sans_extension, disable_reference_audio, character, pitch_factor, auto_tune, \
                reduce_metallic_sound, output_filename_sans_extension, gpu_id, session_id = parse_inputs()
            pitch_options = assemble_pitch_options(auto_tune, reduce_metallic_sound)
            reference_audio = prepare_reference_audio(input_filename_sans_extension, disable_reference_audio, cache,
                                                      session_id)
            link_model_path(character)
            execute_program(user_text, reference_audio, character, pitch_factor, pitch_options, gpu_id)
            copy_output_audio(hsc.get_singleton_file(RESULTS_DIR), output_filename_sans_extension, cache, session_id)
            hsc.clean_up(get_temp_files())
        except BadInputException:
            code = 400
            message = traceback.format_exc()
        except Exception:
            code = 500
            message = hsc.construct_full_error_message(ARCHITECTURE_ROOT, get_temp_files())

        # The message may contain quotes and curly brackets which break JSON syntax, so base64-encode the message.
        message = base64.b64encode(bytes(message, 'utf-8')).decode('utf-8')
        response = {
            "message": message
        }

        return json.dumps(response, sort_keys=True, indent=4), code

    @app.route('/gpu-info', methods=['GET'])
    def get_gpu_info():
        return hsc.get_gpu_info_from_another_venv(PYTHON_EXECUTABLE)

    def parse_inputs():
        schema = {
            'type': 'object',
            'properties': {
                'Inputs': {
                    'type': 'object',
                    'properties': {
                        'User Text': {'type': 'string'},
                        'User Audio': {'type': ['string', 'null']}
                    },
                    'required': ['User Text']
                },
                'Options': {
                    'type': 'object',
                    'properties': {
                        'Architecture': {'type': 'string'},
                        'Disable Reference Audio': {'type': 'boolean'},
                        'Character': {'type': 'string'},
                        'Pitch Factor': {'type': 'integer'},
                        'Auto Tune': {'type': 'boolean'},
                        'Reduce Metallic Sound': {'type': 'boolean'}
                    },
                    'required': ['Disable Reference Audio', 'Character', 'Pitch Factor', 'Auto Tune', 'Reduce Metallic Sound']
                },
                'Output File': {'type': 'string'},
                'GPU ID': {'type': ['string', 'integer']},
                'Session ID': {'type': ['string', 'null']}
            },
            'required': ['Inputs', 'Options', 'Output File', 'GPU ID', 'Session ID']
        }

        try:
            jsonschema.validate(instance=request.json, schema=schema)
        except ValidationError as e:
            raise BadInputException(e.message)

        user_text = request.json['Inputs']['User Text']
        input_filename_sans_extension = request.json['Inputs']['User Audio']
        disable_reference_audio = request.json['Options']['Disable Reference Audio']
        character = request.json['Options']['Character']
        pitch_factor = request.json['Options']['Pitch Factor']
        auto_tune = request.json['Options']['Auto Tune']
        reduce_metallic_sound = request.json['Options']['Reduce Metallic Sound']
        output_filename_sans_extension = request.json['Output File']
        gpu_id = request.json['GPU ID']
        session_id = request.json['Session ID']

        return user_text, input_filename_sans_extension, disable_reference_audio, character, pitch_factor,auto_tune, \
            reduce_metallic_sound, output_filename_sans_extension, gpu_id, session_id

    def assemble_pitch_options(auto_tune, reduce_metallic_sound):
        return [
            *(('pc',) if auto_tune else ()),
            *(('srec',) if reduce_metallic_sound else ()),
        ]

    class BadInputException(Exception):
        pass

    def link_model_path(character):
        """Create a symbolic link to the model folder in the location where Controllable TalkNet expects to find it."""
        character_dir = hsc.character_dir(ARCHITECTURE_NAME, character)
        symlink_dir = os.path.join(ARCHITECTURE_ROOT, 'models', character)
        hsc.create_link(character_dir, symlink_dir)

    def prepare_reference_audio(input_filename_sans_extension, disable_reference_audio, cache, session_id):
        """Temporarily copy the input file to the location where Controllable Talknet expects to find it."""
        if disable_reference_audio or input_filename_sans_extension is None:
            return None
        target = os.path.join(ARCHITECTURE_ROOT, REFERENCE_PREFIX + input_filename_sans_extension + TALKNET_INPUT_EXTENSION)
        try:
            array, samplerate = cache.read_audio_from_cache(Stage.PREPROCESSED, session_id, input_filename_sans_extension)
            soundfile.write(target, array, samplerate, format=TALKNET_INPUT_FORMAT)
        except Exception as e:
            raise Exception("Unable to copy file from Hay Say's audio cache to controllable_talknet's root directory.") \
                from e
        return target

    def execute_program(user_text, reference_audio, character, pitch_factor, pitch_options, gpu_id):
        arguments = [
            '--text', user_text,
            '--custom_model', character,
            *(('--reference_audio', reference_audio) if reference_audio else ()),
            *(('--pitch_factor', str(pitch_factor)) if pitch_factor else ()),
            *(('--pitch_options', *pitch_options) if pitch_options else ())
        ]
        env = hsc.select_hardware(gpu_id)
        subprocess.run([PYTHON_EXECUTABLE, INFERENCE_CODE_PATH, *arguments], env=env)

    def get_temp_files():
        reference_audios = hsc.get_files_with_extension(ARCHITECTURE_ROOT, TALKNET_INPUT_EXTENSION)
        output_files = [os.path.join(RESULTS_DIR, file) for file in os.listdir(RESULTS_DIR)]
        return reference_audios + output_files

    def copy_output_audio(input_path, output_filename_sans_extension, cache, session_id):
        """Copy the output file to the location where Hay Say expects to find it."""
        try:
            array, samplerate = hsc.read_audio(input_path)
            cache.save_audio_to_cache(Stage.OUTPUT, session_id, output_filename_sans_extension, array, samplerate)
        except Exception as e:
            raise Exception("Unable to copy file from Hay Say's audio cache to controllable_talknet's root directory.") \
                from e


def parse_arguments():
    parser = argparse.ArgumentParser(prog='main.py', description='A webservice interface for TTS inference with Controllable Talknet')
    parser.add_argument('--cache_implementation', default='file', choices=hsc.cache_implementation_map.keys(), help='Selects an implementation for the audio cache, e.g. saving them to files or to a database.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    cache = hsc.select_cache_implementation(args.cache_implementation)
    register_methods(cache)
    app.run(debug=True, host='0.0.0.0', port=6574)

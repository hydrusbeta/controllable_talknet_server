from hay_say_common import ROOT_DIR, PREPROCESSED_DIR, OUTPUT_DIR, create_link, get_model_path, clean_up, \
    construct_full_error_message, read_audio_from_cache, read_audio, save_audio_to_cache, get_singleton_file

from flask import Flask, request
import jsonschema
from jsonschema.exceptions import ValidationError

import os
import os.path
import traceback
import json
import base64
import subprocess
import soundfile

ARCHITECTURE_NAME = 'controllable_talknet'
ARCHITECTURE_ROOT = os.path.join(ROOT_DIR, ARCHITECTURE_NAME)

PYTHON_EXECUTABLE = os.path.join(ROOT_DIR, '.venvs', 'controllable_talknet', 'bin', 'python')
INFERENCE_CODE_PATH = os.path.join(ROOT_DIR, 'controllable_talknet', 'command_line_interface.py')

TALKNET_INPUT_FORMAT, TALKNET_INPUT_EXTENSION = 'WAV', '.wav'

app = Flask(__name__)


@app.route('/generate', methods=['POST'])
def generate() -> (str, int):
    code = 200
    message = ""
    # todo: create a get_temp_files() method that can be called from anywhere, instead of keeping track of temp files
    #  as we go
    tmp_files = []
    try:
        user_text, input_filename_sans_extension, character, pitch_factor, pitch_options, \
            output_filename_sans_extension = parse_inputs()
        tmp_input_file = copy_input_audio(input_filename_sans_extension)
        tmp_files = tmp_files + [tmp_input_file] if tmp_input_file else tmp_files
        link_model_path(character)
        execute_program(user_text, input_filename_sans_extension, character, pitch_factor, pitch_options)
        tmp_output_file = get_temp_output_path()
        tmp_files = tmp_files + [tmp_output_file] if tmp_output_file else tmp_files
        copy_output_audio(tmp_output_file, output_filename_sans_extension)
        # write_audio_file(OUTPUT_DIR, output_filename_sans_extension, output_array, output_samplerate)
        clean_up(tmp_files)
    except BadInputException:
        code = 400
        message = traceback.format_exc()
    except Exception:
        code = 500
        message = construct_full_error_message(ARCHITECTURE_ROOT, tmp_files)

    # The message may contain quotes and curly brackets which break JSON syntax, so base64-encode the message.
    message = base64.b64encode(bytes(message, 'utf-8')).decode('utf-8')
    response = {
        "message": message
    }

    return json.dumps(response, sort_keys=True, indent=4), code


def parse_inputs():
    schema = {
        'type': 'object',
        'properties': {
            'Inputs': {
                'type': 'object',
                'properties': {
                    'User Text': {'type': 'string'},
                    'User Audio': {'type': 'string'}
                },
                'additionalProperties': False,
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
                'additionalProperties': False,
                'required': ['Disable Reference Audio', 'Character', 'Pitch Factor', 'Auto Tune', 'Reduce Metallic Sound']
            },
            'Output File': {'type': 'string'}
        },
        'additionalProperties': False,
        'required': ['Inputs', 'Options', 'Output File']
    }

    try:
        jsonschema.validate(instance=request.json, schema=schema)
    except ValidationError as e:
        raise BadInputException(e.Message)

    user_text = request.json['Inputs']['User Text']
    input_filename_sans_extension = request.json['Inputs']['User Audio']
    disable_reference_audio = request.json['Options']['Disable Reference Audio']
    character = request.json['Options']['Character']
    pitch_factor = request.json['Options']['Pitch Factor']
    auto_tune = request.json['Options']['Auto Tune']
    reduce_metallic_sound = request.json['Options']['Reduce Metallic Sound']
    output_filename_sans_extension = request.json['Output File']

    pitch_options = assemble_pitch_options(disable_reference_audio, pitch_factor, auto_tune, reduce_metallic_sound)

    return user_text, input_filename_sans_extension, character, pitch_factor, pitch_options, output_filename_sans_extension


def assemble_pitch_options(disable_reference_audio, pitch_factor, auto_tune, reduce_metallic_sound):
    disable_reference_audio_str = 'dra' if disable_reference_audio else None
    change_input_pitch = 'pf' if pitch_factor != 0 else None
    auto_tune_str = 'pc' if auto_tune else None
    reduce_metallic_sound_str = 'srec' if reduce_metallic_sound else None
    pitch_options = [disable_reference_audio_str, change_input_pitch, auto_tune_str, reduce_metallic_sound_str]
    pitch_options = [option for option in pitch_options if option]  # removes all options that are "None"
    pitch_options = pitch_options if pitch_options else ['']
    return pitch_options


class BadInputException(Exception):
    pass


def link_model_path(character):
    """Create a symbolic link to the model folder in the location where Controllable TalkNet expects to find it."""
    character_dir = get_model_path(ARCHITECTURE_NAME, character)
    symlink_dir = os.path.join(ARCHITECTURE_ROOT, 'models', character)
    create_link(character_dir, symlink_dir)


def copy_input_audio(input_filename_sans_extension):
    """Temporarily copy the input file to the location where Controllable Talknet expects to find it."""
    # todo: make sure the input file does not have the same name as any other file already in CONTROLLABLE_TALKNET_ROOT.
    # todo: make sure we are supplying a format that controllable-talknet can use.
    if input_filename_sans_extension is None:
        return None
    target = os.path.join(ARCHITECTURE_ROOT, input_filename_sans_extension + TALKNET_INPUT_EXTENSION)
    try:
        array, samplerate = read_audio_from_cache(PREPROCESSED_DIR, input_filename_sans_extension)
        soundfile.write(target, array, samplerate, format=TALKNET_INPUT_FORMAT)
    except Exception as e:
        raise Exception("Unable to copy file from Hay Say's audio cache to controllable_talknet's "
                        "root directory.") from e
    return target


def execute_program(user_text, input_filename_sans_extension, character, pitch_factor, pitch_options):
    # todo: redirect stdout to a log file.
    arguments = [
        user_text,
        (input_filename_sans_extension + TALKNET_INPUT_EXTENSION) if input_filename_sans_extension else '-',
        character,
        str(pitch_factor),
        *pitch_options
    ]
    arguments = [argument for argument in arguments if argument]  # Removes all "None" objects from the list
    subprocess.run([PYTHON_EXECUTABLE, INFERENCE_CODE_PATH, *arguments])


def get_temp_output_path():
    directory = os.path.join(ARCHITECTURE_ROOT, 'results')
    path = get_singleton_file(directory)
    return path


def copy_output_audio(path, output_filename_sans_extension):
    """Copy the output file to the location where Hay Say expects to find it."""
    try:
        array, samplerate = read_audio(path)
        save_audio_to_cache(OUTPUT_DIR, output_filename_sans_extension, array, samplerate)
    except Exception as e:
        raise Exception("Unable to copy file from Hay Say's audio cache to controllable_talknet's root directory.") \
            from e


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=6574)

from hay_say_common import ROOT_DIR, PREPROCESSED_DIR, OUTPUT_DIR, create_link, get_model_path, clean_up, \
    construct_full_error_message, read_audio_from_cache, read_audio, save_audio_to_cache, get_singleton_file

from flask import Flask, request

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
        ensure_pretrained_model_links_exist()
        user_text, input_filename_sans_extension, character, pitch_options, output_filename_sans_extension \
            = parse_inputs()
        tmp_input_file = copy_input_audio(input_filename_sans_extension)
        tmp_files = tmp_files + [tmp_input_file] if tmp_input_file else tmp_files
        link_model_path(character)
        execute_program(user_text, input_filename_sans_extension, character, pitch_options)
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


def ensure_pretrained_model_links_exist():
    """Create symbolic links to the pretrained models so that Controllable TalkNet can find them."""
    hifirec_path = get_model_path(ARCHITECTURE_NAME, 'hifirec')
    hifisr_path = get_model_path(ARCHITECTURE_NAME, 'hifisr')
    vqgan_path = get_model_path(ARCHITECTURE_NAME, 'vqgan32_universal_57000.ckpt')
    hifirec_symlink = os.path.join(ARCHITECTURE_ROOT, 'models', 'hifirec')
    hifisr_symlink = os.path.join(ARCHITECTURE_ROOT, 'models', 'hifisr')
    vqgan_symlink = os.path.join(ARCHITECTURE_ROOT, 'models', 'vqgan32_universal_57000.ckpt')
    create_link(hifirec_path, hifirec_symlink)
    create_link(hifisr_path, hifisr_symlink)
    create_link(vqgan_path, vqgan_symlink)


def parse_inputs():
    # todo: add other pitch options, like autotune and adjust input pitch
    check_for_missing_keys()
    user_text = request.json['Inputs']['User Text']
    input_filename_sans_extension = request.json['Inputs']['User Audio']
    character = request.json['Options']['Character']
    disable_text = request.json['Options']['Disable Text']
    output_filename_sans_extension = request.json['Output File']
    check_types(user_text, input_filename_sans_extension, character, disable_text, output_filename_sans_extension)
    pitch_options = ['dra' if disable_text else '']
    return user_text, input_filename_sans_extension, character, pitch_options, output_filename_sans_extension


def check_for_missing_keys():
    missing_user_text = ('Inputs' not in request.json.keys()) or ('User Text' not in request.json['Inputs'].keys())
    missing_user_audio = ('Inputs' not in request.json.keys()) or ('User Audio' not in request.json['Inputs'].keys())
    missing_character = ('Options' not in request.json.keys()) or ('Character' not in request.json['Options'].keys())
    missing_disable_text = ('Options' not in request.json.keys()) \
        or ('Disable Text' not in request.json['Options'].keys())
    missing_output_filename = 'Output File' not in request.json.keys()
    if missing_user_text or missing_user_audio or missing_character or missing_disable_text or missing_output_filename:
        message = ('Missing "User Text" \n' if missing_user_text else '') \
                + ('Missing "User Audio" \n' if missing_user_audio else '') \
                + ('Missing "Character" \n' if missing_character else '') \
                + ('Missing "Disable Text" \n' if missing_disable_text else '') \
                + ('Missing "Output File" +n' if missing_output_filename else '')
        raise BadInputException(message)


def check_types(user_text, user_audio, character, disable_text, output_filename):
    wrong_type_user_text = not isinstance(user_text, str)
    wrong_type_user_audio = not isinstance(user_audio, str)
    wrong_type_character = not isinstance(character, str)
    wrong_type_disable_text = not isinstance(disable_text, bool)
    wrong_type_output_filename = not isinstance(output_filename, str)
    if wrong_type_user_text or wrong_type_user_audio or wrong_type_character or wrong_type_disable_text \
            or wrong_type_output_filename:
        message = ('"User Text" should be a string \n' if wrong_type_user_text else '') \
                + ('"User Audio" should be a string \n' if wrong_type_user_audio else '') \
                + ('"Character" should be a string \n' if wrong_type_character else '') \
                + ('"Disable Text" should be an bool \n' if wrong_type_disable_text else '') \
                + ('"Output File" should be a string \n' if wrong_type_output_filename else '')
        raise BadInputException(message)


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
    target = os.path.join(ARCHITECTURE_ROOT, input_filename_sans_extension + TALKNET_INPUT_EXTENSION)
    try:
        array, samplerate = read_audio_from_cache(PREPROCESSED_DIR, input_filename_sans_extension)
        soundfile.write(target, array, samplerate, format=TALKNET_INPUT_FORMAT)
    except Exception as e:
        raise Exception("Unable to copy file from Hay Say's audio cache to controllable_talknet's "
                        "root directory.") from e
    return target


def execute_program(user_text, input_filename_sans_extension, character, pitch_options):
    # todo: redirect stdout to a log file.
    subprocess.run([PYTHON_EXECUTABLE, INFERENCE_CODE_PATH,
                    user_text, input_filename_sans_extension + TALKNET_INPUT_EXTENSION, character, *pitch_options])


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

"""
utilities
"""

import random
import datetime

def unique_filename(
    extension,
    mode='timestamp',
    base=None,
    position='base_first'
):
    """
    creates unique filenames, using methods corresponding to the 'mode' argument.

    # TODO: add more methods

    argument
        extension
            The extension of the file
        mode
            The method of creating a unique identification
        base
            The base name of the filename
            If None, the identification will be the filename
        position
            The position of the base filename
            Only used when 'base' is passed to the function

    return
        filename
            The created unique file name
    """
    mode_PATTERN = [
        'random',
        'timestamp',
    ]

    if not mode in mode_PATTERN:
        raise Exception('Mode name {} not available. Chose from {}.'.format(mode, mode_PATTERN))
    elif mode == mode_PATTERN[0]:
        unique_id = str(random.randint(0, 99999))
    elif mode == mode_PATTERN[1]:
        unique_id = datetime.datetime.now().strftime('%Hh%Mm%Ss%fms')

    if not '.' in extension:
        extension = '.' + extension

    filename = unique_id + extension

    if base:
        position_PATTERN = [
            'base_first',
            'unique_last',
            'base_last',
            'unique_first'
        ]

        if not position in position_PATTERN:
            raise Exception('Position {} not available. Chose from {}.'.format(position, position_PATTERN))
        elif position == position_PATTERN[0] or position == position_PATTERN[1]:
            filename = '_'.join([base, unique_id]) + extension
        elif position == position_PATTERN[2] or position == position_PATTERN[3]:
            filename = '_'.join([unique_id, base]) + extension

    return filename

import glob

def get_available_serial_port(prefix='/dev/tty.usb'):
    """
    Lists serial port names that start with the given prefix.

    Args:
        prefix (str): The prefix of serial ports to search for, e.g., '/dev/tty.USB'

    Returns:
        list: A list of available serial ports matching the prefix
    """
    # macOS는 /dev/tty.* /dev/cu.* 사용
    # Linux는 /dev/ttyUSB* /dev/ttyACM* 사용
    pattern = prefix + "*"

    ports = glob.glob(pattern)
    ports.sort()

    return ports[0]
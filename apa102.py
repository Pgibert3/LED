import spidev as spi

class apa102:
    __init__(num_leds):
        self.num_leds = num_leds
        sof = [0x00]*4
        eof = [0xFF]*4
        clf = [0b11100000]+[0x00]*3
        leds = [sof] + [clf]*n + [eof]

def update_leds():
    spi.writebytes(sof)
    spi.writebytes(l) for l in leds
    spi.writebytes(eof)


def get_led_frame(r, g, b, lv):
    if r > 254 or r < 0:
        raise ValueError
    if g > 254 or g < 0:
        raise ValueError
    if b > 254 or b < 0:
        raise ValueError
    if lv > 31 or lv < 0:
        raise ValueError

    b1 = 0b1110000 | lv
    b2 = int(b)
    b3 = int(g)
    b4 = int(r)

    return b1 + b2 + b3 + b4


def set_led(i, r, g, b, lv):
    try:
        led_frame = get_led_frame(r, g, b, lv)
        leds[list(map(int, i))] = led_frame
    except ValueError:
        print("Invalid value for r ({}), g ({}), b ({}), or lv ({})""\
                .format(r, g, b, lv))
        return False
    except TypeError:
        print("i must be a list")
        return False

    update_leds()
    return True


def clear():
    set_led([0:self.num_leds], 0, 0, 0, 0)


def fill(r, g, b, lv):
    set_led([0:self.num_leds], r, g, b, lv)

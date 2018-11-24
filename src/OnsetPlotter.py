import matplotlib.pyplot as plt
import numpy as np
from OnsetStrengthFilter import OnsetStrengthFilter  # TODO: Setup project structure in PyCharm

SAMPLE_RATE = 44100
HOP_LENGTH = 512
OS_BUFFER_SIZE = 2500
N_FFT = 2048
N_MELS = 128
NUM_SUBBANDS = 8

OSFilter = OnsetStrengthFilter(SAMPLE_RATE, HOP_LENGTH, OS_BUFFER_SIZE, N_FFT, N_MELS,NUM_SUBBANDS)


# adds os subbands to the envelope and returns to new envelope
def append_os_env(os_bands, os_env):
    return np.append(os_env, os_bands, axis=0)


# computes the x axis of os subband data in units of seconds
def get_time_axis(os_env_length, hop_length, sr):
    tot_time = (os_env_length * hop_length) / sr
    return np.linspace(0, tot_time, os_env_length)


# main loop
def main():
    os_env = np.zeros((1, NUM_SUBBANDS))
    print("Reading audio...")
    while True:
        try:
            os_bands = OSFilter.next_os_all_bands()
            os_env = append_os_env(os_bands, os_env)
        except KeyboardInterrupt:
            break

    # plot the result
    print("Done")
    print("Plotting os vs time")
    os_env = np.delete(os_env, 0, axis=0)  # remove default value
    x = get_time_axis(os_env.shape[0], HOP_LENGTH, SAMPLE_RATE)
    plt.plot(x, os_env)
    plt.show()


if __name__ == "__main__":
    main()

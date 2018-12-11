import argparse
import numpy as np
import matplotlib.pyplot as plt
from os import path, mkdir
import pickle
import io

from librosa import mel_frequencies

from OnsetStrengthFilter import OnsetStrengthFilter


def env_append(env, item):
    """
    Appends item to env and returns env
    @param env: array getting appended
    @type env: np.ndarray((x,y))
    @param item: item getting appended to array
    @type item: np.ndarray(y)
    @return: appended env
    @rtype: np.ndarray((x+1,y))
    """
    env = np.append(env, [item], axis=0)
    return env


def get_time_axis(num_os_samples, hop_length, sr):
    """
    Computes the x axis of an audio sample in units of seconds
    @param num_os_samples: number of os samples collected
    @type num_os_samples: int
    @param hop_length: hop length between frames
    @type hop_length: int
    @param sr: sampling rate
    @type sr: int
    @return: time series
    @rtype: np.ndarray(num_os_samples * hop_length / sr)
    """
    tot_time = (num_os_samples * hop_length) / sr
    return np.linspace(0, tot_time, num_os_samples)


def post_format_env(env):
    """
    Removes the first sample from all subbands of env and transposes
    @param env: the envelope to format
    @type env: np.ndarray((x,y))
    @return: formatted envelope
    @rtype: np.ndarray((y,x-1))
    """
    env = np.delete(env, 0, axis=0)
    env = np.transpose(env)
    return env


def main():
    # argparse setup for terminal app
    parser = argparse.ArgumentParser()

    parser.add_argument("subbands",
                        help="number of os subbands to plot",
                        action="store",
                        type=int)
    parser.add_argument("-o", "--output",
                        help="specify an .png file to output the resulting plot to",
                        action="store",
                        type=str)
    parser.add_argument("-t", "--trails",
                        help="setting this flag plots the trailing average and max (from peak picker) of each os subband",
                        action="store_true",
                        default=False)
    parser.add_argument("-O", "--onsets",
                        help="plot onsets according to the local onset engine, the librosa engine, or both",
                        action="store",
                        type=str,
                        choices=["local", "librosa", "both"])

    args = parser.parse_args()

    # audio sampling Setup
    SR = 44100
    HOP_LEN = 512
    OS_BUFF_SIZE = 2500
    NFFT = 2048
    NMELS = 128
    NUM_SUBBANDS = 4

    os_filter = OnsetStrengthFilter(SR, HOP_LEN, OS_BUFF_SIZE, NFFT, NMELS, NUM_SUBBANDS)
    os_env = np.zeros((1, NUM_SUBBANDS))  # stores os values of each subband
    onset_env = np.zeros((1, NUM_SUBBANDS))  # stores onsets of each subband
    max_env = np.zeros((1, NUM_SUBBANDS))  # stores running previous max values of each subband
    avg_env = np.zeros((1, NUM_SUBBANDS))  # stores running previous average of each subband

    # record audio
    input("Press Enter to start recording")
    print("recording...")
    while True:
        try:
            bands = os_filter.next_os_all_bands()
            env_data = os_filter.is_os_peak(bands, debug=True)  # TODO: extract all kwargs of is_os_peak()

            # update envelopes
            os_env = env_append(os_env, bands)
            onset_env = env_append(onset_env, env_data["peaks"])
            max_env = env_append(max_env, env_data["prev_max"])
            avg_env = env_append(avg_env, env_data["prev_avg"])

        except KeyboardInterrupt:
            print("done recording")
            break

    # format envelopes
    os_env = post_format_env(os_env)
    onset_env = post_format_env(onset_env)
    max_env = post_format_env(max_env)
    avg_env = post_format_env(avg_env)

    # plotting
    # setup
    fig = plt.figure()
    axs = []
    rows = NUM_SUBBANDS
    cols = 1

    freqs = mel_frequencies(n_mels=NMELS) # Used for determining the frequency ranges of each subband
    x_axis = get_time_axis(np.shape(os_env)[1], HOP_LEN, SR)

    # initialize axes
    for i in range(0, NUM_SUBBANDS):
        ax = fig.add_subplot(rows, cols, i+1)
        lower_freq_index = max(0, int((i * len(freqs) / NUM_SUBBANDS)) - 1)
        upper_freq_index = int(lower_freq_index + (len(freqs)/NUM_SUBBANDS))
        ax.set_title("Band%d (%d-%dHz)" % (i, freqs[lower_freq_index], freqs[upper_freq_index]))
        ax.set_xlabel("time (seconds)")
        ax.set_ylabel("magnitude")
        axs.append(ax)

    # plot os_env
    for i in range(0, NUM_SUBBANDS):

        axs[i].plot(x_axis, os_env[i], color="royalblue", linewidth=1.5, label="os")

    # plot trails
    if args.trails:
        # plot max_env
        for i in range(0, NUM_SUBBANDS):
            axs[i].plot(x_axis, max_env[i], color="orange", linewidth=1, label="trailing os max")

        # plot avg_env
        for i in range(0, NUM_SUBBANDS):
            axs[i].plot(x_axis, avg_env[i], color="green", linewidth=1, label="trailing os avg")

    # plot local onsets
    if args.onsets == "local" or args.onsets == "both":
        for i_b in range(0, NUM_SUBBANDS):
            for i_e in range(0, np.shape(onset_env)[1]):
                if onset_env[i_b, i_e]:
                    axs[i_b].axvline(x=x_axis[i_e], color="red")

    # apply legend
    for ax in axs:
        ax.legend(loc="upper right")

    fig.tight_layout()  # space subplots

    # save files
    if args.output is not None:
        mkdir(args.output)
        fig.savefig(path.join(args.output, "all_bands.png"))

        # save individual subplots
        for i in range(0, len(axs)):
            # Generate a file name
            lower_freq_index = max(0, int((i * len(freqs) / NUM_SUBBANDS)) - 1)
            upper_freq_index = int(lower_freq_index + (len(freqs) / NUM_SUBBANDS))
            fname = "band%d_%d-%dHz.png" % (i, freqs[lower_freq_index], freqs[upper_freq_index])

            # pickle axes data from fig into sub_fig
            buf = io.BytesIO()
            pickle.dump(fig, buf)
            buf.seek(0)
            sub_fig = pickle.load(buf)

            # delete all axes but the one we want
            for a, ax in enumerate(sub_fig.axes):
                if a != i:
                    sub_fig.delaxes(ax)

            # use a temporary subplot to reset the axes grid of sub_fig
            temp = sub_fig.add_subplot(111)
            sub_fig.axes[0].set_position(temp.get_position())
            temp.remove()

            sub_fig.savefig(path.join(args.output, fname))

    plt.show()


if __name__ == "__main__":
    main()



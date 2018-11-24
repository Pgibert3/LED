#Configure settings for reading audio input and extracting features

project = {
    "SAMPLE_RATE" : 44100 #Sample rate of project
}

input_settings = {
        "sr" : 44100,
        "frame_size" : 2048,
        "hop_length" : 512,
}

extractor_settings = {
        "channels" : [0,16,32,128],
        "sr" : 44100,
        "hop_length" : 512,
        "n_fft" : 2048,
        "oss_buff_length" : 2500,
}

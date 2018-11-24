#Configure settings for reading audio input and extracting features

project = {
    "SAMPLE_RATE" : 44100 #Sample rate of project
}

input = {
    "FRAME_SIZE" : 2048, #Total samples per frame
    "HOP_LENGTH" : 512 #How many new samples to read into a frame
}

extractor = {
    "N_FFT" : 2048, #Total frames per analysis buffer for feature of onset_strength_multi()
    "HOP_LENGTH" : 512, #New frames read in per analysis buffer. See N_FFT
    "OSS_CHANNELS" : [0, 4, 32, 64, 128],
    "OSS_BUFF_LENGTH" : 2500
}

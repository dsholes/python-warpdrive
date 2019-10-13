# Warpdrive

Sync audio from multiple sources. Uses Dynamic Time Warping (DTW) to measure delay between two sources of audio, pads audio with silence to align (writing new audio files), and outputs a JSON file with info about delay between sources. Warpdrive is heavily dependent on the awesome `librosa` library.


NOTES and WARNINGS:
- Rough first cut, structure of code will change dramatically in the future
- Right now, new audio files are being compressed. Need to find new way of writing (especially stereo files) while maintaining quality of original audio
    - Can use JSON output to "nudge" original audio in REAPER or DAW of choice to align, instead of using "padded" audio for now.
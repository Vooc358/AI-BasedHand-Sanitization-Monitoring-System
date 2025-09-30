from gtts import gTTS
import os

# Text to be converted to speech
text = "Please follow the above steps for proper sanitization"
print(text)

# Create a gTTS object
tts = gTTS(text=text, lang='en', slow=False)

# Save the audio file
audio_file = "sanitization_instructions.mp3"
tts.save(audio_file)

# Play the audio (for Windows, macOS, or Linux)
if os.name == 'nt':  # For Windows
    os.system(f'start {audio_file}')
elif os.name == 'posix':  # For macOS or Linux
    os.system(f'open {audio_file}')  # For macOS


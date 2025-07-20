import pyttsx3

# Intialize TTS engine
engine = pyttsx3.init()
engine.setProperty('rate',engine.getProperty('rate')-50)

def speak(text):
    engine.say(text)
    engine.runAndWait()
import speech_recognition as sr

def speech_to_text():
    # Initialize the recognizer
    recognizer = sr.Recognizer()

    # Use the default microphone as the source
    with sr.Microphone() as source:
        print("Listening...")

        # Adjust for ambient noise if necessary
        recognizer.adjust_for_ambient_noise(source)

        # Listen for speech
        audio = recognizer.listen(source)

        print("Processing...")

        # Recognize speech using Google Speech Recognition
        try:
            # Use recognizer.recognize_google() for Google's service
            # You can also try other recognizers like recognizer.recognize_sphinx() for offline recognition
            text = recognizer.recognize_google(audio)
            print("You said:", text)
            return text
        except sr.UnknownValueError:
            print("Sorry, I could not understand what you said.")
        except sr.RequestError as e:
            print("Could not request results from Google Speech Recognition service; {0}".format(e))

# Example usage
speech_to_text()

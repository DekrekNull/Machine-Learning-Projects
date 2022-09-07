import os
import FortuneCookieClassifier as cookie
import OCRClassifier as OCR

OUT_FILE ="./data/output.txt"

# Entry funtion for the algorithm
def main():
    os.remove(OUT_FILE)
    cookie.cookie_classifier()
    OCR.OCR_classifier()

if __name__ == '__main__':
	main()
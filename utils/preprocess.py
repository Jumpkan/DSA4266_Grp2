from bs4 import BeautifulSoup
from deep_translator import GoogleTranslator
import re
from unidecode import unidecode
import string
import pandas as pd
import email
from langdetect import detect
import warnings
from bs4 import MarkupResemblesLocatorWarning
warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

class DataPreProcessor():
    def __init__(self):
        self.translator = GoogleTranslator(source='auto', target='en')

    def html_remover(self, doc):
        """
        Removes html tags in a given text
        Input: doc
        Output: String
        """
        txt = doc["pipe_text"]
        soup=BeautifulSoup(txt,'html.parser')
        a=soup.get_text()
        doc["pipe_text"] = a
        return doc

    # char limit 5000 currently 100000

    def illegal_char_remover(self, doc):
        """
        Removes some illegal characters that are not in unicode (represented in hex or bytes).
        """
        txt = doc["pipe_text"]
        txt_encoded = txt.encode("unicode_escape")
        txt_encoded_cleaned = re.sub(b'\\\\x[a-f0-9][a-f0-9]', b'', txt_encoded)
        txt_cleaned = txt_encoded_cleaned.decode("unicode_escape")
        doc["pipe_text"] = txt_cleaned
        return doc

    #Additions by Josiah

    def replace_tokens(self, doc):
        '''
        Replaces links, money, ip, numbers
        '''
        text = doc["pipe_text"]
        # Replace links with token [link]
        text = re.sub("http\S+", " [link] ", text)
        # Replace money with token [MONEY]
        text = re.sub(r"[$]\d+[.,]*\d*", " [money] ", text)
        text = re.sub(r"\d+[.,]*\d*[$]", " [money] ", text)
        # Replace ip addresses with token [ip]
        text = re.sub(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$", " [ip] ", text)
        # Replace emails with token [email]
        text = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b", " [email] ", text)
        # Remove words with a number in the middle, like s8d or 7a8s8r. Does not remove office365 or 250k
        text = re.sub(r"\d*(([a-zA-Z]+)(\d+))+[a-zA-Z]+\d*", " ", text)
        # Replace numbers with a special token [NUM]
        text = re.sub(r"[\d]+", " [num] ", text)
        # Standardise whitespace
        text = re.sub('[\s]+'," ", text)
        # lowercase text
        text = text.lower()
        doc["pipe_text"] = text
        return doc
    
    def remove_nonenglish(self, doc):
        '''
        Removes characters from other languages
        '''
        text = doc["pipe_text"]
        text = re.sub(r'[^a-zA-Z0-9\]!"$%&\'()*+,./:;=#@?[\\^_`{|}~\-[\u00C0-\u01DA]+', " ", text) #Remove characters from other languages, except diacritics
        text = unidecode(text) # strip accents
        doc["pipe_text"] = text
        return doc
    
    def remove_punctuations(self, doc):
        text = doc["pipe_text"]
        text = text.translate(str.maketrans('', '', string.punctuation))
        doc["pipe_text"] = text
        return doc
        
    def join_spaces(self, text):
        '''
        Joins single letters that are separated by a single space, e.g w a n t  m o r e -> want more
        '''
        split = text.split(" ")
        ans = []
        temp = []
        for part in split:
            if len(part) == 1:
                temp.append(part)
            else:
                if len(temp)>0:
                    word = ''.join(temp)
                    ans.append(word)
                temp = []
                if len(part) != 0:
                    ans.append(part)
        word = ''.join(temp)
        ans.append(word)
        ans = " ".join(ans)
        return ans

    def clean_unicode(self, doc):
        '''
        Attempts to remove nonsensical characters and convert letterlike symbols back to letters.
        '''
        text = doc["pipe_text"]
        words = []
        text = str(text)
        text = text.replace("\n", "  ")
        text = text.replace(chr(160)," ") # Replace weird space
        text = re.sub(r"[\u2800-\u28ff]+", "", text)
        text = self.join_spaces(text)
        for word in text.split(" "):
            if len(word) == 0:
                continue
            if unidecode(word) == "":
                continue
            code = [ord(char) for char in word]
            if max(code)>65535: # Outside Basic multilangual plane
                word = unidecode(word)
            words.append(word)
        text = " ".join(words)
        doc["pipe_text"] = text
        return doc

    def detect_lang(self, doc):
        '''
        Tries to detect the language of text, reducing number of text that has to be passed through translator.
        Note that it is not accurate at detecting the actual language, just weeds out English text.
        '''
        text = doc["pipe_text"]
        text = re.sub("[link]", "", text)
        text = re.sub(r"[!'#$%&\"()*+,-./:;<=>?@[\\\]^_`{|}~]+", " ", text)
        text = re.sub(r"[0-9]+", " ", text)
        if len(text) == 0:
            doc["lang"] = None
        try:
            lang = detect(text)
        except:
            text = unidecode(text)
            try:
                lang = detect(text)
            except:
                print(text)
                doc["lang"] = None
                return doc
        doc["lang"] = lang
        return doc

    def translate(self, doc, char_limit=50000):
        """
        Can only deal with one string (not a list)
        """
        text = doc["pipe_text"]
        if doc.get("lang") == "en":
            doc["translated"] = text
            doc["pipe_text"] = text
            return doc
        # Limit the translation to 50000 chars. If larger, dont translate.
        if len(text) <= char_limit:

            if len(text) <= 1000:
                try:
                    translated = self.translator.translate(text)
                except Exception as e:
                    print(e)
                    translated = text
                if translated==None:
                    translated = ""
                doc["translated"] = translated
                doc["pipe_text"] = doc["translated"]
                return doc
            
            # Split text into chunks that the translation engine can handle. We limit this to 4000 char as of now
            words = doc["pipe_text"].split(" ")
            chunks = []
            char_count = 0
            temp_chunk = []
            for word in words:
                char_count += (len(word) + 1)
                if char_count <= 1000:
                    temp_chunk.append(word)
                else:
                    chunks.append(" ".join(temp_chunk))
                    char_count = len(word)
                    temp_chunk = [word]

            if len(temp_chunk) > 0:
                chunks.append(" ".join(temp_chunk))

            new_pipe_text = []
            for chunk in chunks:
                if len(chunk) > 1000:
                    translated_text = chunk
                else:
                    try:
                        translated_text = self.translator.translate(chunk)
                    except Exception as e:
                        print(e)
                        translated_text = chunk
                    if not translated_text:
                        translated_text = ''
                new_pipe_text.append(translated_text)

            full_translated = ' '.join(new_pipe_text)
        else:
            full_translated = text
        if full_translated == None:
            full_translated = ""
        doc["translated"] = full_translated
        doc["pipe_text"] = full_translated
        return doc
    
 
    def preprocess(self, text, first_pipe, second_pipe, doc_char_limit=5000):
        """
        The returned doc might have the following fields, depending on the pipes
        - original_text
        - translated
        - sents
        - pipes is a list of lists of various pipe components 
            E.g. [[html_remover, illegal_char_remover, translator], [url_remover, consec_newliine_remover]]
        """

        # Limit the number of characters if not there will be a memory error
        doc = {
            "original_text": text,
            "pipe_text": text,
        }

        # Sentencizer has to be the last pipe.
        pipe_component_to_func = {
            "html_remover": self.html_remover,
            "illegal_char_remover": self.illegal_char_remover,
            "translator": self.translate,
            "clean_unicode": self.clean_unicode,
            "detect_lang": self.detect_lang,
        }
        for pipe_component in first_pipe:
            doc = pipe_component_to_func[pipe_component](doc)
        if not doc.get("translated"):
            doc["translated"] = doc["pipe_text"]
        # Truncate pipe text and then do the remaining operations
        doc["first_pipe_text"] = doc["pipe_text"]
        doc["pipe_text"] = doc["pipe_text"][:doc_char_limit]
        second_pipe_component_to_func = {
            "remove_nonenglish": self.remove_nonenglish,
            "remove_punctuation": self.remove_punctuations,
            "replace_tokens": self.replace_tokens,
        }

        for pipe_component in second_pipe:
            doc = second_pipe_component_to_func[pipe_component](doc)
        doc["second_pipe_text"] = doc["pipe_text"]
        doc.pop("pipe_text")
        return doc
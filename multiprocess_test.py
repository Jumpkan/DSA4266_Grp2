import multiprocessing
import logging
import json
import pandas as pd
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning
from deep_translator import GoogleTranslator
import re
from unidecode import unidecode
import string
import email
from langdetect import detect
import warnings
warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

## Helper functions

# Pre-translation processing
def html_remover(txt):
    """
    Removes html tags in a given text
    Input: doc
    Output: String
    """
    soup=BeautifulSoup(txt,'html.parser')
    a=soup.get_text()
    return a

def join_spaces(text):
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

def clean_unicode(text):
    '''
    Attempts to remove nonsensical characters and convert letterlike symbols back to letters.
    '''
    words = []
    text = str(text)
    text = text.replace("\n", "  ")
    text = text.replace(chr(160)," ") # Replace weird space
    text = re.sub(r"[\u2800-\u28ff]+", "", text)
    text = join_spaces(text)
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
    return text

def detect_lang(text):
    '''
    Tries to detect the language of text, reducing number of text that has to be passed through translator.
    Note that it is not accurate at detecting the actual language, just weeds out English text.
    '''
    text = re.sub("[link]", "", text)
    text = re.sub(r"[!'#$%&\"()*+,-./:;<=>?@[\\\]^_`{|}~]+", " ", text)
    text = re.sub(r"[0-9]+", " ", text)
    if len(text) == 0:
        return None
    try:
        lang = detect(text)
    except:
        text = unidecode(text)
        try:
            lang = detect(text)
        except:
            print(text)
            return None
    return lang

def translate(text, translator, max_size=1000):
    '''
    Breaks up large texts into chunks for translation
    '''
    words = text.split(" ")
    chunks = []
    char_count = 0
    temp_chunk = []
    for word in words:
        char_count += (len(word) + 1)
        if char_count <= max_size:
            temp_chunk.append(word)
        else:
            chunks.append(" ".join(temp_chunk))
            char_count = len(word)
            temp_chunk = [word]

    if len(temp_chunk) > 0:
        chunks.append(" ".join(temp_chunk))

    new_pipe_text = []
    for chunk in chunks:
        if len(chunk) > max_size:
            translated_text = chunk
        else:
            try:
                translated_text = translator.translate(chunk)
            except Exception as e:
                print(e)
                translated_text = chunk
            if not translated_text:
                translated_text = ''
        new_pipe_text.append(translated_text)
    full_translated = ' '.join(new_pipe_text)
    return full_translated

# Post translation processing
def replace_tokens(text):
    '''
    Replaces links, money, ip, numbers
    '''
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
    return text

def remove_nonenglish(text):
    '''
    Removes characters from other languages
    '''
    text = re.sub(r'[^a-zA-Z0-9\]!"$%&\'()*+,./:;=#@?[\\^_`{|}~\-[\u00C0-\u01DA]+', " ", text) #Remove characters from other languages, except diacritics
    text = unidecode(text) # strip accents
    return text

def remove_punctuations(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

# Processing pipeline

def dataframe_generator(full_data, output_queue, batch_size=100):
    num_rows = len(full_data)
    start = 0
    while start < num_rows:
        end = min(start + batch_size, num_rows)
        subset_data = full_data[start:end]
        batch_data = list(zip(list(subset_data.index), list(subset_data.message)))
        output_queue.put(batch_data)
        # Move the start index for the next iteration
        start = end
    output_queue.put(None)

def pre_translation_preprocessing_worker(document):
    # Perform preprocessing on document
    doc_id, message = document
    message = html_remover(message)
    message = clean_unicode(message)
    lang = detect_lang(message)
    if lang and lang!="en":
        message_length = len(message)
    else:
        message_length = 0
    return (doc_id, message, lang, message_length)

def batch_translator(input_queue, output_queue, max_length=1000):
    batch_ids = []
    batch_messages = []
    batch_length = 0
    auto_translator = GoogleTranslator(source='auto', target='en')
    while True:
        item = input_queue.get()
        if item is None:  # Check for the sentinel value
            if batch_messages:  # Ensure any remaining batch gets sent
                try:
                    translated_messages = auto_translator.translate_batch(batch_messages)
                except:
                    print(batch_messages)
                    translated_messages = batch_messages
                full_outputs = zip(batch_ids, translated_messages)
                for item in full_outputs:
                    output_queue.put(item)
            output_queue.put(None)  # Signal the consumer this is the end
            break
        doc_id, message, lang, message_length = item
        if not lang or lang=="en":
            output_queue.put((doc_id, message))
        elif message_length > max_length:
            translated_message = translate(message, auto_translator)
            output_queue.put((doc_id, translated_message))
        else:
            new_batch_length = batch_length + message_length
            if new_batch_length >= max_length:
                # Translate as a batch
                try:
                    translated_messages = auto_translator.translate_batch(batch_messages)
                except:
                    print(batch_messages)
                    translated_messages = batch_messages
                full_outputs = zip(batch_ids, translated_messages)
                for item in full_outputs:
                    # (doc_id, translated_message)
                    output_queue.put(item)
                # Reset batch
                batch_ids = []
                batch_messages = []
                batch_length = 0
            # Add new item
            batch_ids.append(doc_id)
            batch_messages.append(message)
            batch_length += message_length

def post_translation_processing_worker(output_queue, output_file):
    with open(output_file, "a") as f:  # Use append mode
        while True:
            item = output_queue.get()
            if item is None:  # End of processing
                break
            doc_id, translated_message = item
            if not translated_message:
                # In case message is None
                translated_message = ""
            processed_message = remove_nonenglish(translated_message)
            processed_message = replace_tokens(processed_message)
            processed_message = remove_punctuations(processed_message)
            document = {"doc_id": doc_id, "translated": translated_message, "processed": processed_message}
            json.dump(document, f)
            f.write("\n")    
        

def preprocess_pipeline(input_documents, output_file, max_length=1000):
    # Use Manager queue here for better stability in some systems
    manager = multiprocessing.Manager()
    pre_translation_input_queue = manager.Queue()
    pre_translation_output_queue = manager.Queue()
    output_queue = manager.Queue()
    # Prepare processes
    data_generator_process = multiprocessing.Process(target=dataframe_generator, args=(input_documents, pre_translation_input_queue, 100))
    translation_process = multiprocessing.Process(target=batch_translator, args=(pre_translation_output_queue, output_queue, max_length))
    post_translation_process = multiprocessing.Process(target=post_translation_processing_worker, args=(output_queue, output_file))
    data_generator_process.start()
    translation_process.start()
    post_translation_process.start()

    # Distribute documents to preprocessing
    with multiprocessing.Pool(processes=multiprocessing.cpu_count() // 2) as pool:
        data_batch = pre_translation_input_queue.get()
        while data_batch is not None:
            result = pool.map(pre_translation_preprocessing_worker, data_batch)
            for item in result:
                pre_translation_output_queue.put(item)
            data_batch = pre_translation_input_queue.get()
    pre_translation_output_queue.put(None)  # Signal the end to batcher

    translation_process.join()
    post_translation_process.join()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    input_documents = pd.read_pickle("Data/sample.pkl")
    output_file = "output.pkl"
    max_length = 1000
    with open(output_file, "w") as f: # Creates output file
        pass
    preprocess_pipeline(input_documents, output_file, max_length)
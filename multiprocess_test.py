import multiprocessing
import logging
import queue
import json

def dataframe_generator(full_data, output_queue, batch_size=100):
    num_rows = len(full_data)
    start = 0
    while start < num_rows:
        end = min(start + batch_size, num_rows)
        batch_data = full_data[start:end]
        output_queue.put(batch_data)
        # Move the start index for the next iteration
        start = end
    output_queue.put(None)

def pre_translation_preprocessing_worker(document):
    # Perform preprocessing on document
    processed_document = document  # Placeholder for processing
    return processed_document

def batcher(input_queue, output_queue, batch_size):
    batch = []
    while True:
        item = input_queue.get()
        if item is None:  # Check for the sentinel value
            if batch:  # Ensure any remaining batch gets sent
                output_queue.put(batch)
            output_queue.put(None)  # Signal the consumer this is the end
            break
        batch.append(item)
        if len(batch) >= batch_size:
            output_queue.put(batch)
            batch = []

def post_translation_processing_worker(output_queue, output_file):
    with open(output_file, "a") as f:  # Use append mode
        while True:
            batch = output_queue.get()
            if batch is None:  # End of processing
                break
            for item in batch:
                document = {"data": item}
                json.dump(document, f)
                f.write("\n")

def preprocess_pipeline(input_documents, output_file, batch_size):
    # Use Manager queue here for better stability in some systems
    manager = multiprocessing.Manager()
    pre_translation_input_queue = manager.Queue()
    pre_translation_output_queue = manager.Queue()
    output_queue = manager.Queue()
    # Prepare processes
    data_generator_process = multiprocessing.Process(target=dataframe_generator, args=(input_documents, pre_translation_input_queue, 100))
    translation_process = multiprocessing.Process(target=batcher, args=(pre_translation_output_queue, output_queue, batch_size))
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
    input_documents = list(range(10000))  # Example list of input documents
    output_file = "output.jsonl"
    with open(output_file, "w") as f: # Creates output file
        pass
    preprocess_pipeline(input_documents, output_file, batch_size=10)
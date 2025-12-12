import openai

def get_embeddings_batch(text_list, model="text-embedding-3-small", batch_size=100):
    
    if len(text_list) <= batch_size:
        response = openai.embeddings.create(input=text_list, model=model)
        return [embedding.embedding for embedding in response.data]
    
    all_embeddings = []
    counter = 1
    for i in range(0, len(text_list), batch_size):
        batch = text_list[i:i + batch_size]
        response = openai.embeddings.create(input=batch, model=model)
        all_embeddings.extend([embedding.embedding for embedding in response.data])
        print(f"Processed {counter * batch_size} of {len(text_list)}")
        counter += 1
    
    return all_embeddings
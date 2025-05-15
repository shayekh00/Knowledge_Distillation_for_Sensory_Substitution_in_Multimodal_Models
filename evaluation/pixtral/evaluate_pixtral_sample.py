from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration
model_id = "mistral-community/pixtral-12b"
model = LlavaForConditionalGeneration.from_pretrained(model_id).to("cuda")
processor = AutoProcessor.from_pretrained(model_id)

IMG_URLS = [
"https://picsum.photos/id/237/400/300", 
"https://picsum.photos/id/231/200/300", 
"https://picsum.photos/id/27/500/500",
"https://picsum.photos/id/17/150/600",
]
PROMPT = "<s>[INST]Describe the images.\n[IMG][IMG][IMG][IMG][/INST]"

inputs = processor(text=PROMPT, images=IMG_URLS, return_tensors="pt").to("cuda")
generate_ids = model.generate(**inputs, max_new_tokens=500)
output = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print(output)
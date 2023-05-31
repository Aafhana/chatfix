#from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration
from transformers import AutoTokenizer, BartForConditionalGeneration
#import youtube_transcript_api
from youtube_transcript_api import YouTubeTranscriptApi
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
#mname = "facebook/blenderbot-400M-distill"
# model = BlenderbotForConditionalGeneration.from_pretrained(mname)
# tokenizer = BlenderbotTokenizer.from_pretrained(mname)

from transformers import T5ForConditionalGeneration, T5Tokenizer
model_name = 't5-base'
model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name)

import openai
openai.api_key="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"

# with open("C:/Users/afree/Desktop/etxt_summ.txt",'r',encoding='utf-8') as f:
#     file=f.read()
#print('hello')



def chat(uinput):
    inputs = tokenizer([uinput], return_tensors="pt")
    reply_ids = model.generate(**inputs)
    chat_resp=tokenizer.batch_decode(reply_ids)
    #print(tokenizer.batch_decode(reply_ids))
    #chat_resp.encoding
    return chat_resp


def chatt5(file):
    inputs = tokenizer.encode("summarize: " +file, return_tensors="pt", max_length=1000, truncation=True)
    #print(inputs)
# Generate the summary
    outputs = model.generate(inputs, max_length=150, num_beams=4, early_stopping=True)
#print(outputs)
# Decode the generated summary
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary

def summary_api(url):
    
    #url = request.args.get('url', '')
    video_id = url.split('.be/')[1]
   #video_id='https://youtu.be/vzLmRomOP4Q'
    summary = sum_gpt(get_transcript(video_id))
    return summary
    #return summary, 200

def get_transcript(video_id):
    transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
    transcript = ' '.join([d['text'] for d in transcript_list])
    return transcript
    
    
def summarizer(file):
    summary = ''
    for i in range(0, (len(file)//1000)+1):
        inputs = tokenizer([file[i*1000:(i+1)*1000]], max_length=1024,max_new_tokens=1024, return_tensors="pt",truncation=True)
        summary_ids = model.generate(inputs["input_ids"], num_beams=4,min_length=30, max_length=40)
        summary_text=tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        summary = summary + summary_text + ' '
        
    return summary

def sum_gpt(file):
    response = openai.Completion.create(
    engine='text-davinci-003',
    prompt='Summarize the following text in 5 lines:' +str(file),
    max_tokens=120)

    summary = response.choices[0].text.strip()
    return summary
    

    


# while 1:
#      uinput= input()
#      if uinput=='quit':
#          exit(0)
#      y=chat(uinput)
#      print(type(y))
#      print(y)  
input=('''The C language shook the computer world. Its impact should not be
underestimated, because it fundamentally changed the way programming was
approached and thought about. The creation of C was a direct result of the need
for a structured, efficient, high-level language that could replace assembly code
when creating systems programs. As you may know, when a computer language
is designed, trade-offs are often made, such as the following:
• Ease-of-use versus power
• Safety versus efficiency
• Rigidity versus extensibility
Prior to C, programmers usually had to choose between languages that
optimized one set of traits or the other. For example, although FORTRAN
could be used to write fairly efficient programs for scientific applications, it
was not very good for system code. And while BASIC was easy to learn, it
wasnt very powerful, and its lack of structure made its usefulness questionable
for large programs. Assembly language can be used to produce highly efficient
programs, but it is not easy to learn or use effectively. Further, debugging
assembly code can be quite difficult.''')
#print(summarizer(input))
# y=summary_api('https://youtu.be/vzLmRomOP4Q')
# print(y)
 
 
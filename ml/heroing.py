from pywebio.input import *;
from pywebio.output import *;

code = textarea("Enter code", code={
    'mode' : "python",
    'theme' : 'darcula'
},value="import")
password = input("Input password",type=PASSWORD);

gift = select("what do you want",['keyboard','ipad']);

agree = checkbox("User term",["Agree"])

answer = radio("choose one",options=['A','b'])

upload = file_upload("Upload the file")


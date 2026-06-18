text = "The name is sameer and phone number is 7989344950";

found = 'name' in text;

print(found)

import re;

pattern = 'phone';

found2 = re.search(pattern,text);

print(found2.span());

(x,y) = found2.span();

print(x,y);

__doc__;


import re
import ast

def is_valid_time_window_list(s):
    pattern = r'^\[\[(\d+,\s*\d+)(,\s*\[\d+,\s*\d+\])*\]\]$'
    a = re.findall(r'(\d+)', s)
    print(a)
    if not re.match(pattern, s):
        return False
    
    try:
        lst = ast.literal_eval(s)
        
        if not isinstance(lst, list):
            return False
        
        for item in lst:
            if not isinstance(item, list) or len(item) != 2:
                return False
            start, end = item
            if not (isinstance(start, int) and isinstance(end, int)):
                return False
            if start > end:
                return False
            
        return True
    except (ValueError, SyntaxError):
        return False
    
a = is_valid_time_window_list('AMan00 55, 58, 60], [60, 700], [102, 122], seconds')
print(a)
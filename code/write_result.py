import os
def write_result(results, output_dir, filename):
    #result format: [["text",[result]], ["text", [result]], ... ]
    
    output_file = os.path.join(output_dir,filename)
    f= open(output_file,"w")
    for i in results:
        text = i[0]
        content = i[1]
        f.write(text + ": "+ ", ".join(map(str,content)) + "\n")
    f.close()


def write_data(data, output_dir, filename):
    #result format: [["text",[result]], ["text", [result]], ... ]
    
    output_file = os.path.join(output_dir,filename)
    f= open(output_file,"w")
    for i in data:
        f.write( ", ".join(map(str,i)) + "\n")
    f.close()


with open("all_videos.txt", "r+") as f:
    text = f.readlines()
    f.seek(0)
    newtext = []
    for line in text:
        # print(line)
        newtext.append(line.split()[0][:-4] + "\n")
    f.writelines(newtext)
    f.truncate()

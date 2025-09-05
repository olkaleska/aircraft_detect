"""
Працюємо з файлами, чистимо від зайвої інфи
тобто копіюєм інфу з Опен Стріт Мепс
і ця штука нам переведе її в чітке csv зь лише потрібною інфою
"""

def filter_csv(file_input, file_output):
    """
    Ця функція витягує тільки потрібні дані з усіх доступних
    """
    res = []
    with open(file_input, "r", encoding="UTF-8") as file:
        print('njklnjklm')
        # print(file)
        content = file.read()
        print("Довжина content:", len(content))  # скоріш за все буде 0
        # print(content.encode("utf-8", errors="replace").decode("utf-8"))
        data = content.split('<way id')
        # print(data)
            # data = data.split('<way id')
        for objectt in data:
            # print(objectt)
            objectt = objectt.split('\n')
            line_list = []
            for line in objectt:
                
                if "<center lat" in line:
                    # <center lat="47.2733628" lon="39.6405598"/>
                    # print(line)
                    x = line.split('="')[1].split('"')[0]
                    x = "x: " + x
                    y = line.split('="')[2][:-3]
                    y = ", y: " + y
                    line_list.append(x + y)
                    # print(1)
                elif '<tag k="military"' in line:
                    # <tag k="military" v="airfield"/>
                    line = line.split('v="')[1][:-3]
                    line_list.append(line)
                    # print(2)
                elif '<tag k="name"' in line:
                    # <tag k="name" v="Ростов-на-Дону — Центральный"/>
                    line = line.split('v="')[1][:-3]
                    line_list.append(line)
                    # print(3)
                elif '<tag k="name:en"' in line:
                    # <tag k="name:en" v="Rostov-on-Don North Air Base"/>
                    line = line.split('v="')[1][:-3]
                    line_list.append(line)
                    # print(4)
                elif '<tag k="official_status"' in line:
                    # <tag k="official_status" v="ru:авиабаза"/>
                    line = line.split('v="')[1][:-3]
                    line_list.append(line)
                    # print(5)
                elif '<tag k="old_official_status"' in line:
                    # <tag k="old_official_status" v="ru:международный аэропорт"/>
                    line = line.split('v="')[1][:-3]
                    line_list.append(line)
                    # print(6)
                # print(line)
            # print(line_list)
            line_list = ','.join(line_list)
            print(line_list)
            res.append(line_list)
        res = '\n'.join(res)
        # print(res)
        with open(file_output, "w", encoding="UTF-8") as data:
            data.write(res)




if __name__=="__main__":
    filter_csv('./files_with_coords/coords_befor_01.txt', './files_with_coords/coords_01.txt')

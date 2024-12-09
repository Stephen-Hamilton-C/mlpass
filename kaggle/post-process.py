with open("kaggle/data.csv", "r") as data_file:
    with open("kaggle/data-processed.csv", "w") as output_file:
        for line in data_file:
            # Recreate the header file to easily concat
            if line.strip() == "password,strength":
                output_file.write("Password,Strength_Level\n")
                continue

            line_split = line.split(",")
            if len(line_split) > 2:
                line = line_split[-2] + "," + line_split[-1]
                line_split = line_split[-2:]

            # Map categories
            if line_split[1].strip() == "1":
                line_split[1] = "2\n"
                line = line_split[0] + "," + line_split[1]
            elif line_split[1].strip() == "2":
                line_split[1] = "4\n"
                line = line_split[0] + "," + line_split[1]

            output_file.write(line)

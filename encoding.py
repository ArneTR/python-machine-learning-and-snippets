# reading a datastream and guessing encoding
import chardet
with open("./PATH_TO_FILE/FILE.csv", 'rb') as rawdata:
    result = chardet.detect(rawdata.read(10000))

print(result)

# then the file can be read:
pd.read("./PATH_TO_FILE/FILE.csv", encoding=result["encoding"])


# Case 2: You have a malformatted string
## First decode with the broken encoding you think that caused the problem
## Then you have it in byte format. From here, encode to UTF-8

my_str = "j849thghndskfh783r fj83qhjr83"

decoded = my_str.decode("windows-1251")

decoded.encode("utf-8")

# Also the other way round may help, because you always can encode as utf-8 and decode from there.
# encode just writes the chars as their representation in the charset.
# so if a string was originally iso-8859-1 then interpreted as utf-8 you have to: iso_string.encode("iso-8859-1").decode()

import socket

# NEVER do this:
    # sock = socket.socket()
    # sock.connect((address, port))
# if anything fails the socket will be left unclosed.
# Either use a try / catch at least, or safely use the with statement
# Important: The with-block internally does the try/catch for us and always
# runs the __exit__ function of the called block. In this case 
# the __exit__ of socket.socket() which will close()

with socket.socket() as sock:
    sock.connect(("datafuse.de", 22))
    print(sock.recv(1024))
print("Done and closed automatically")

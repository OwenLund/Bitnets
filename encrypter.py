from Crypto.Cipher import AES
from Crypto import Random
import os, random, struct

class crypt:
	def __init__(self):
		self.key = open(os.path.join(os.path.expanduser('~'),'BitnetsAESKey.txt'), "r").read()

	def encrypt(self, filename):
		chunksize=64*1024
		out_filename = filename + '.enc'
		iv = ''.join(chr(random.randint(0, 0xFF)) for i in range(16))
		encryptor = AES.new(self.key, AES.MODE_CBC, iv)

		filesize = os.path.getsize(filename)
		with open(filename, 'rb') as infile:
			with open(out_filename, 'wb') as outfile:
				outfile.write(struct.pack('<Q', filesize))
				outfile.write(iv)

			 	while True:
					chunk = infile.read(chunksize)
					if len(chunk) == 0:
						break
					elif len(chunk) % 16 != 0:
						chunk += ' ' * (16 - len(chunk) % 16)
					outfile.write(encryptor.encrypt(chunk))
		os.remove(filename)

	def decrypt(self, filename):
		chunksize=24*1024
		encoded_file = filename + ".enc"
		with open(encoded_file, 'rb') as infile:
			origsize = struct.unpack('<Q', infile.read(struct.calcsize('Q')))[0]
			iv = infile.read(16)
			decryptor = AES.new(self.key, AES.MODE_CBC, iv)

			with open(filename, 'wb') as outfile:
				while True:
					chunk = infile.read(chunksize)
					if len(chunk) == 0:
						break
					outfile.write(decryptor.decrypt(chunk))

				outfile.truncate(origsize)
		os.remove(encoded_file)

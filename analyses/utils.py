def montar_divisor(texto, tamanho, simbolo):
	print("\n" + simbolo * tamanho)
	print(" " * ((tamanho - len(texto)) // 2) + texto)
	print(simbolo * tamanho + "\n")
#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import matplotlib.pyplot as plt
import soundfile as sf
from scipy import signal

# Para detectar sincronismo.
from scipy import signal as sig
import random
import pandas as pd

import app_decoder


# In[2]:


def texttobin(input_string):
    bit_string=[]
    ascii_array=bytearray(input_string,'ascii')
    for i in ascii_array:
        bin_char=bin(i)
        byte=bin_char[2:]
        byte_size=len(byte)
        while(byte_size<8):
            byte='0'+byte
            byte_size=len(byte)
        for bit in byte:
            bit_string=np.append(bit_string,bit)
    return bit_string


# In[3]:


# leitura de txt (header):

header = open('header.txt', 'r', encoding="utf-8")
header = header.read()
header = header.replace("[", "")
header = header.replace("]", "")
header = header.replace(",", "")

header = np.fromstring(header, dtype=int, sep=' ')
print(header)
print(type(header[2]))
print(len(header))


# In[4]:


# debug

print(len(header))
print(header)
type(header[2])


# In[5]:


print("Digite o número de modulações que o detector irá detectar (2 ou 4): ")
n = input()
n = int(n)


# In[6]:


print("Digite a taxa de transmissão de preferência (Tb): ")
baudRate = input()
baudRate = int(baudRate)


# # 2-FSK (se n = 2)

# In[7]:


if n == 2:

    # Leitura de .wav
    
    data, Fs = sf.read('teste_2fsk.wav') 
    print("Dados do áudio: ", data)
    print("Tamanho dos dados: ", len(data))
    print("Sample rate of the audio: ", Fs)
    
    
    # Cálculo de up sample (amostras por símbolo).
    
    up_sample = Fs/baudRate
    print("Amostras por símbolo: ", up_sample)
    
    
    # modulação do header.
    
    F1=800
    F2=1200
    t_wave=np.arange(0,1/baudRate,1/Fs)         # Limita meu sinal em um tempo determinado.
    
    header_modulado = []
    tam = len(header)
    cosseno_F1 = np.cos(2*np.pi*F1*t_wave)
    cosseno_F2 = np.cos(2*np.pi*F2*t_wave)
    
    for i in range(0, tam, 1):
        
        if header[i]==0:
            x = (cosseno_F1)
            header_modulado = np.append(header_modulado, x)
        elif header[i]==1:
            x = (cosseno_F2)
            header_modulado = np.append(header_modulado, x)
            
    print(header_modulado)
    len(header_modulado)
    
    
    # Plot do header modulado (não deu para ver a mudança de frequência).
    
    t = np.arange(0,len(header_modulado),1)  
    fig = plt.figure()
    plt.plot(t, header_modulado)
    fig.savefig('plot.png')
    header_modulado
    
    
    # Aqui eu tenho que descobrir onde o sinal começa, ou seja, achar o cabeçalho e ver onde sinal começa (start_bit). 
    # Isso pode ser feito com uma função de correlação (o valor deve ser máximo para o atraso estar sincronizado).
    
    corr = np.correlate(data,header_modulado,mode='valid')
    corr_position = np.argmax(corr)
    start_bit = corr_position
    
    
    # Plot da correlação cruzada
    
    plt.plot(corr)
    plt.title('Identificação do início do cabeçalho')
    plt.plot(corr_position,corr[corr_position],'or')
    plt.show()
    
    
    # Start bit só vamos saber depois de fazer o sincronismo.
    
    print("Amostra de começo da mensagem: ", start_bit)
    print(len(header_modulado))
    
    
    # Cálculos para plotar o sinal recebido. (não é necessário, nem decisivo para a demodulação)
    
    start = start_bit + len(header_modulado)    # ACHO que o start_bit dá a primeira posição do header. Para achar
                                                 # onde a mensagem começa, tenho que somar isso ao tamanho do 
                                                    # header. Se der errado tenta do outro jeito.
            
    
    # Pegar dados a partir de onde começa a mensagem.
    
    data = data[start:]
    print(data)
    print(len(data))
    
    
    # debug
    
    print(baudRate, 10/baudRate)
    print(Fs, int((10/baudRate)*Fs))
    print(data)
    len(data)
    
    
    # Construção das ondas da portadora do 2-FSK. Cada "ramo" com uma frequência.
    
    F1=800
    F2=1200
    t_wave=np.arange(0,1/baudRate,1/Fs)
    
    wave1=np.cos(2*np.pi*F1*t_wave)
    wave2=np.cos(2*np.pi*F2*t_wave)
    
    
    # debug
    
    print(wave2)
    print(np.flip(wave2))       # Precisa-se inverter pq o filtro casado é h(-t).
    print(len(wave2))
    
    
    # Aplicação do filtro casado + média móvel.
    
    casado_1=np.convolve(data,np.flip(wave1))
    casado_1=np.convolve(np.abs(casado_1),np.ones((int(len(t_wave)/2))))
    casado_2=np.convolve(data,np.flip(wave2))
    casado_2=np.convolve(np.abs(casado_2),np.ones((int(len(t_wave)/2))))
    
    
    # debug
    
    print("Tamanho do data: ", len(data))
    print("Tamanho do vetor após convolução: ", len(casado_1))
    
    
    # Amostragem do filtro casado a cada período de bit. Pego só última amostra para representar meu símbolo.
    # Por causa da média móvel e da envoltória, esse ponto que eu pegar vai ser meu símbolo.
    
    t = np.arange(0,len(data)/Fs,1/Fs)
    
    step=int(Fs/baudRate)
    
    amostra_casado1=casado_1[step::step]
    amostra_casado2=casado_2[step::step]
    
    t_amostra=np.arange(step/Fs,t[-1]+step/Fs,step/Fs)
    len(t_amostra)
    
    
    # Plot da saída dos filtros casados.
    
    t = np.arange(0,len(data)/Fs,1/Fs) 
    
    fig, axs = plt.subplots(2)
    fig.suptitle('Saída dos filtros casados')
    axs[0].plot(t,casado_1[0:len(t)])
    axs[0].plot(t_amostra,amostra_casado1[0:len(t_amostra)],'or')
    axs[1].plot(t,casado_2[0:len(t)])
    axs[1].plot(t_amostra,amostra_casado2[0:len(t_amostra)],'or')
    plt.show()
    fig.savefig('saida_casada.png')
    
    
    # Plot da saída dos filtros casados. COM ZOOM.
    
    t = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    t_amostra = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] 
    
    fig, axs = plt.subplots(2)
    fig.suptitle('Saída dos filtros casados com apenas algumas amostras')
    axs[0].plot(t,casado_1[10:20])
    axs[0].plot(t_amostra,amostra_casado1[10:20],'or')
    axs[1].plot(t,casado_2[10:20])
    axs[1].plot(t_amostra,amostra_casado2[10:20],'or')
    plt.show()
    
    
    # debug
    
    print(step)
    amostra_casado2
    
    
    # Coparação posição a posição do vetor de saída de cada filtro.
    
    a1 = len(amostra_casado1)
    a2 = len(amostra_casado2)
    vetor_final = []
    
    for i in range(0, a1, 1):
        if amostra_casado1[i]>amostra_casado2[i]:
            vetor_final.append(0)
        elif amostra_casado1[i]<amostra_casado2[i]:
            vetor_final.append(1)
            
    app_decoder.app_decoder(vetor_final)


# # 4FSK ( se n = 4)

# In[10]:


if n == 4:

    # Leitura de .wav
    
    data, Fs = sf.read('teste_4fsk.wav') 
    print("Dados do áudio: ", data)
    print("Tamanho dos dados: ", len(data))
    print("Sample rate of the audio: ", Fs)
    
    
    # Portadoras da 4-FSK
    
    F1=600
    F2=800
    F3=1000
    F4=1200
    
    t_wave=np.arange(0,1/baudRate,1/Fs)
    
    wave1=np.cos(2*np.pi*F1*t_wave)
    wave2=np.cos(2*np.pi*F2*t_wave)
    wave3=np.cos(2*np.pi*F3*t_wave)
    wave4=np.cos(2*np.pi*F4*t_wave)
    
    #Apresentação das portadoras
    fig, axs = plt.subplots(2,2,constrained_layout=True)
    fig.suptitle('Portadoras 4-FSK')
    axs[0,0].plot(t_wave,wave1)
    axs[0,0].set_title('Portadora do símbolo 00 - 600 Hz')
    axs[0,1].plot(t_wave,wave2)
    axs[0,1].set_title('Portadora do símbolo 01 - 800 Hz')
    axs[1,0].plot(t_wave,wave3)
    axs[1,0].set_title('Portadora do símbolo 10 - 1000 Hz')
    axs[1,1].plot(t_wave,wave4)
    axs[1,1].set_title('Portadora do símbolo 11 - 1200 Hz')
    plt.show()
    
    
    # Modulação do header. No caso do 4fsk, a gente precisa agrupar de dois em dois bits para representar cada 
    # simbolo. Depois fazemos uma modulação.
    
    #F1=600
    #F2=800                                     
    #F3=1000                                    
    #F4=1200
    #t_wave=np.arange(0,1/baudRate,1/Fs)         # Limita meu sinal em um tempo determinado.
    #
    #b = [1]
    #header = np.append(header,b)
    #
    #header_modulado = []
    #tam = len(header)
    #cosseno_F1 = np.cos(2*np.pi*F1*t_wave)
    #cosseno_F2 = np.cos(2*np.pi*F2*t_wave)
    #cosseno_F3 = np.cos(2*np.pi*F3*t_wave)
    #cosseno_F4 = np.cos(2*np.pi*F4*t_wave)
    #
    #for i in range(0, tam, 2):
    #    
    #    if header[i]==0 and header[i+1]==0:
    #        x = (cosseno_F1)
    #        header_modulado = np.append(header_modulado, x)
    #    elif header[i]==0 and header[i+1]==1:
    #        x = (cosseno_F2)
    #        header_modulado = np.append(header_modulado, x)
    #    elif header[i]==1 and header[i+1]==1:
    #        x = (cosseno_F3)
    #        header_modulado = np.append(header_modulado, x)
    #    elif header[i]==1 and header[i+1]==0:
    #        x = (cosseno_F4)
    #        header_modulado = np.append(header_modulado, x)
    #        
    #print(header_modulado)
    #len(header_modulado)
    
    
    # Aqui eu tenho que descobrir onde o sinal começa, ou seja, achar o cabeçalho e ver onde sinal começa (start_bit). 
    # Isso pode ser feito com uma função de correlação (o valor deve ser máximo para o atraso estar sincronizado).
    
    #corr = np.correlate(data,header_modulado,mode='valid')
    #corr_position = np.argmax(corr)
    #start_bit = corr_position
    
    
    #Plot da correlação cruzada
    
    #plt.plot(corr)
    #plt.title('Identificação do início do cabeçalho')
    #plt.plot(corr_position,corr[corr_position],'or')
    #plt.show()
    
    
    # Start bit só vamos saber depois de fazer o sincronismo.
    
    #print("Bit de começo da mensagem: ", start_bit)
    
    
    # Pegar dados a partir de onde começa a mensagem.
    
    #start = start_bit + len(header_modulado) - 600
    #data = data[start:]
    #print(data)
    #print(len(data))
    
    
    # Aplicação do filtro casado + média móvel.
    
    casado_1=np.convolve(data,np.flip(wave1))
    casado_1=np.convolve(np.abs(casado_1),np.ones((int(len(t_wave)/2))))
    
    casado_2=np.convolve(data,np.flip(wave2))
    casado_2=np.convolve(np.abs(casado_2),np.ones((int(len(t_wave)/2))))
    
    casado_3=np.convolve(data,np.flip(wave3))
    casado_3=np.convolve(np.abs(casado_3),np.ones((int(len(t_wave)/2))))
    
    casado_4=np.convolve(data,np.flip(wave4))
    casado_4=np.convolve(np.abs(casado_4),np.ones((int(len(t_wave)/2))))
    
    
    #Amostragem do filtro casado a cada período de bit
    
    t = np.arange(0,len(data)/Fs,1/Fs)
    
    step=int(Fs/baudRate)
    
    amostra_casado1=casado_1[step::step]
    amostra_casado2=casado_2[step::step]
    amostra_casado3=casado_3[step::step]
    amostra_casado4=casado_4[step::step]
    t_amostra=np.arange(step/Fs,t[-1]+step/Fs,step/Fs)
    
    
    # debug
    
    len(amostra_casado1)      # esse valor nos dá o número de símbolos.
    
    
    # Plot da saída dos filtros casados.
    
    fig, axs = plt.subplots(4)
    fig.suptitle('Saída dos filtros casados')
    axs[0].plot(t,casado_1[0:len(t)])
    axs[0].plot(t_amostra,amostra_casado1[0:len(t_amostra)],'or')
    axs[1].plot(t,casado_2[0:len(t)])
    axs[1].plot(t_amostra,amostra_casado2[0:len(t_amostra)],'or')
    axs[2].plot(t,casado_3[0:len(t)])
    axs[2].plot(t_amostra,amostra_casado3[0:len(t_amostra)],'or')
    axs[3].plot(t,casado_4[0:len(t)])
    axs[3].plot(t_amostra,amostra_casado4[0:len(t_amostra)],'or')
    plt.show()
    
    
    # Saída dos filtros casados, mas com ZOOM.
    
    t = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    t_amostra = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    fig, axs = plt.subplots(4)
    fig.suptitle('Saída dos filtros casados')
    axs[0].plot(t,casado_1[0:10])
    axs[0].plot(t_amostra,amostra_casado1[0:10],'or')
    axs[1].plot(t,casado_2[0:10])
    axs[1].plot(t_amostra,amostra_casado2[0:10],'or')
    axs[2].plot(t,casado_3[0:10])
    axs[2].plot(t_amostra,amostra_casado3[0:10],'or')
    axs[3].plot(t,casado_4[0:10])
    axs[3].plot(t_amostra,amostra_casado4[0:10],'or')
    plt.show()
    
    
    # Comparação posição a posição do vetor de saída de cada filtro.
    
    a1 = len(amostra_casado1)
    vetor_final4 = []
    
    for i in range(0, a1, 1):
        if amostra_casado1[i]>amostra_casado2[i] and amostra_casado1[i]>amostra_casado3[i] and amostra_casado1[i]>amostra_casado4[i]:        
            vetor_final4.append(0)
            vetor_final4.append(0)
        elif amostra_casado2[i]>amostra_casado1[i] and amostra_casado2[i]>amostra_casado3[i] and amostra_casado2[i]>amostra_casado4[i]:
            vetor_final4.append(0)
            vetor_final4.append(1)
        elif amostra_casado3[i]>amostra_casado1[i] and amostra_casado3[i]>amostra_casado2[i] and amostra_casado3[i]>amostra_casado4[i]:
            vetor_final4.append(1)
            vetor_final4.append(1)
        elif amostra_casado4[i]>amostra_casado1[i] and amostra_casado4[i]>amostra_casado2[i] and amostra_casado4[i]>amostra_casado3[i]:
            vetor_final4.append(1)
            vetor_final4.append(0)
            
    #print(vetor_final4)
    
    
    print(type(vetor_final4))
    vetor_final4 = list(vetor_final4)
    print(len(vetor_final4))
    
    
    # Aqui eu tenho que descobrir onde o sinal começa, ou seja, achar o cabeçalho e ver onde sinal começa (start_bit). 
    # Isso pode ser feito com uma função de correlação (o valor deve ser máximo para o atraso estar sincronizado).
    
    corr = np.correlate(vetor_final4,header,mode='valid')
    corr_position = np.argmax(corr)
    start_bit = corr_position
    print("Posição de início do cabeçalho: ", start_bit)
    
    #Plot da correlação cruzada
    
    plt.plot(corr)
    plt.title('Identificação do início do cabeçalho')
    plt.plot(corr_position,corr[corr_position],'or')
    plt.show()
    
    
    start = start_bit + len(header)
    vetor_final4 = vetor_final4[start:]
    
    print("Tamanho do vetor do 4-FSK", len(vetor_final4))
    print(vetor_final4)
    app_decoder.app_decoder(vetor_final4)


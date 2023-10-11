import socket
import json
import torch
from c_plain_test import TestC
from parser1 import Parser

def main(args):
    testc=TestC(args)
    # 创建客户端套接字
    sk = socket.socket()           
    # 尝试连接服务器
    sk.connect(('127.0.0.1',8898))
    print("start_client")
    q_embss,dataset,q_imgs=testc.load_query_data()
    startsig='1'
    ret = sk.recv(4096)
    hash_tab=json.loads(ret.decode('utf-8'))
    print("C:rec_hash_tab")
    embs=[]
    i=0
    while True:
        if i==58:
            break
        emb=q_embss[i].tolist()
        qemb_byte=json.dumps(emb)
        print(qemb_byte,"qemb_byte")
        print("C:sending query emb",i)
        sk.send(bytes(qemb_byte,encoding='utf-8'))
        label=sk.recv(1024)
        label.decode('utf-8')
        print(label,"label")
        i=i+1
    # sk.close() 


if __name__ == '__main__':
    print("emb128")
    main(Parser().parse())
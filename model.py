import torch 
import torch.nn as nn
import math
class InputEmbedding(nn.Module):
    def __init__(self,d_model,vocab_size):
        super().__init__()
        self.d_model=d_model
        self.vocab_size=vocab_size
        self.embedding=nn.Embedding(vocab_size,d_model)
    def forward(self,x):
        return self.embedding(x)*math.sqrt(self.d_model)
class PositionEncoding(nn.Module):
    def __init__(self,d_model,seq_len,dropout):
        super().__init__()
        self.d_model=d_model
        self.seq_len=seq_len
        self.dropout=nn.Dropout(dropout)

        pe=torch.zeros(seq_len,d_model)
        position=torch.arange(0,seq_len,dtype=torch.float).unsqueeze(1)
        div_term=torch.exp(torch.arange(0, d_model, 2).float() *
            (-math.log(10000.0) / d_model))
        pe[:,0::2]=torch.sin(position*div_term)
        pe[:,1::2]=torch.cos(position*div_term)
        pe=pe.unsqueeze(0)
        self.register_buffer('pe',pe)
    def forward(self,x):
        x=x+(self.pe[:,:x.shape[1],:]).requires_grad(False)
        return self.dropout(x)
        
class LayerNormalization(nn.Module):
    def __init__(self,eps:float=10**-6):
        super().__init__()
        self.eps=eps
        self.alpha=nn.parameter(torch.ones(1))
        self.bias=nn.parameter(torch.zeros(1))
    def forward(self,x):
        mean=x.mean(dim=-1,keepdim=True)
        std=x.std(dim=-1,keepdim=True)
        return self.alpha*(x-mean)/(std+self.eps)+self.bias

class FeedForwardNetwork(nn.Module):
    def __init__(self,d_model,d_ff,dropout):
        super().__init__()
        self.d_model=d_model
        self.d_ff=d_ff
        self.dropout=dropout
        self.linear1=nn.Linear(d_model,d_ff) #w1 and b1
        self.dropout=nn.Dropout(dropout)
        self.linear2=nn.Linear(d_ff,d_model)#w2 and b2
    def forward(self,x):
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))
    
class MultiHeadattention(nn.Module):
    def __init__(self,head,d_model,dropout):
        super().__init__()
        self.head=head
        self.d_model=d_model
        
        assert d_model%head==0
        self.d_k=d_model%head
        self.w_q=nn.Linear(d_model,d_model)
        self.w_k=nn.Linear(d_model,d_model)
        self.w_v=nn.Linear(d_model,d_model)
        self.w_o=nn.Linear(d_model,d_model)
        self.dropout=nn.Dropout(dropout)
    @staticmethod
    def attention(query,key,value,mask,dropout:nn.Dropout):
        d_k=query.shape[-1]
        attention_score=(query@ key.transpose(-2,1))//math.sqrt(d_k)
        if mask is not None:
            attention_score.masked_fill_(mask==0,-1e9)
        attention_score=attention_score.softmax(dim=-1)
        if dropout is not None:
            attention_score=dropout(attention_score)
        return (attention_score@value),attention_score
    def forward(self,q,k,v,mask):
        query=self.w_q(q)
        key=self.w_k(k)
        value=self.w_v(v)

        query=query.view(query.shape[0],query.shape[1],self.head,self.d_k).transpose(1,2)
        key=key.view(key.shape[0],key.shape[1],self.head,self.d_k).transpose(1,2)
        value=value.view(value.shape[0],value[1],self.head,self.d_k).transpose(1,2)
        x,self.attention=MultiHeadattention.attention(query,key,value)
        x=x.transpose(1,2).contiguous().view(x.shape[0],-1,self.h*self.d_k)
        return self.w_o(x)
    
class Residualconnection(nn.Module):
    def __init__(self,dropout:float):
        super().__init__()
        self.dropout=nn.Dropout()
        self.norm=LayerNormalization()
    
    def forward(self,x,sublayer):
        return x+self.dropout(sublayer(self.norm(x)))

class EncoderBlock(nn.Module):
    def __init__(self,self_attention,feed_forward_network,dropout):
        super().__init()
        self.self_attention=MultiHeadattention
        feed_forward_network=FeedForwardNetwork
        self.residual_connection=Residualconnection(dropout)
        self.residual_connection2=Residualconnection(dropout)
    def forward(self,x,src_mask):
        x=self.residual_connection(x,lambda x:self.self_attention(x,x,x,src_mask))
        x=self.residual_connection2(x,self.feed_forward_network)
        return x


class Encoder(nn.Module):
    def __init__(self,layers:nn.ModuleList):
        super().__init__()
        self.layers=layers
        self.norm=LayerNormalization()
    def forward(self,x,mask):
        for layer in self.layers:
            x=layer(x,mask)
        return self.norm(x)


class DecoderBlock(nn.Module):
    def __init__(self,self_attention,cross_attention,feed_forward_network,dropout):
        super().__init__()
        self.self_attention=self_attention
        self.cross_attention=cross_attention
        self.feed_forward_network=feed_forward_network
        self.residual_connection=nn.Module(Residualconnection(dropout) for _ in range(3))

    def forward(self,x,encoder_output,src_mask,tgt_mask):
        x=self.residual_connection[0](x,lambda x: self.self_attention(x,x,x,tgt_mask))
        x=self.residual_connection[1](x,lambda x:self.cross_attention(x,encoder_output,encoder_output,src_mask))
        x=self.residual_connection[2](x,self.feed_forward_network)
        return x
class Decoder(nn.Module):
    def __init__(self,layers:nn.ModuleList):
        super().__init__()
        self.layers=layers
        self.norm=LayerNormalization()
    
    def forward(self,x,encoder_output,src_mask,tgt_mask):
        for layer in self.layers:
            x=layer(x,encoder_output,src_mask,tgt_mask)
        return self.norm(x)
    
class projectionlayer(nn.Module):
    def __init__(self,d_model,vocab_size):
        super().__init__()
        self.proj=nn.Linear(d_model,vocab_size)
    def forward(self,x):
        return torch.log_softmax(self.proj(x),dim=-1)
class Transformer(nn.Module):
    def __init__(self,encoder,decoder,src_embedding,tgt_embedding,src_pos,tgt_pos,projection_layer):
        super().__init__()
        self.encoder=encoder
        self.decoder=decoder
        self.src_embedding=src_embedding
        self.tgt_embedding=tgt_embedding
        self.src_pos=src_pos
        self.tgt_pos=tgt_pos
        self.projection_layer=projection_layer

def encoder(self,src,src_mask):
    src=self.src_embedding(src)
    src=self.src_pos(src)
    return self.encoder(src,src_mask)

def decoder(self,encoder_output,src_mask,tgt,tgt_mask):
    tgt=self.tgt_embed(tgt)
    tgt=self.tgt_pos(tgt)
    return self.decoder(tgt,encoder_output,src_mask,tgt_mask)

def project(self,x):
    return self.project_layer(x)

def build_transformer(src_vocav_size,tgt_vocab_size,src_seq_len,tgt_seq_len,d_model:int=512,N:int=6,h:int=8,dropout:float=0.1,d_ff:int=2048):
    #create embeddinglayer
    src_embed=InputEmbedding(d_model,src_vocav_size)
    tgt_embed=InputEmbedding(d_model,tgt_vocab_size)

    #create postion encoding
    src_pos=PositionEncoding(d_model,src_seq_len,dropout)
    tgt_pos=PositionEncoding(d_model,tgt_seq_len,dropout)

    #create encoder block
    encoder_block=[]
    for i in range(N):
        encoder_self_attention_block=MultiHeadattention(d_model,h,dropout)
        feed_forward_block=FeedForwardNetwork(d_model,src_seq_len,dropout)
        encoder_block=EncoderBlock(encoder_self_attention_block,feed_forward_block,dropout)
        encoder_block.append(encoder_block)
    
    #decoder block
    decoder_block=[]
    for i in range(N):
        decoder_self_attention=MultiHeadattention(d_model,h,dropout)
        decoder_cross_attebntion=MultiHeadattention(d_model,h,dropout)
        feed_forward_block=FeedForwardNetwork(d_model,d_ff,dropout)
        decoder_block=DecoderBlock(decoder_self_attention,decoder_cross_attebntion,feed_forward_block,dropout)
        decoder_block.append(decoder_block)


    encoder=Encoder(nn.ModuleList(encoder_block))
    decoder=Decoder(nn.ModuleList(decoder_block))


    projection_layer=projectionlayer(d_model,tgt_vocab_size)

    transformer=Transformer(encoder,decoder,src_embed,tgt_embed,src_pos,tgt_pos,projection_layer)


    #initalize parameter
    for p in transformer.parameter():
        if p.dim() >1:
            nn.init.xavier_uniform(p)
    return transformer







        








    








    











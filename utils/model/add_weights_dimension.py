import torch

orig_t1d  = 22
#orig_t1d  = 23
orig_t2d  = 44
orig_dtor = 30

new_t1d   = orig_t1d + 2 + 4 + 1 #(+2 time step and seq confidence) + 4 (dssp) + 1 (hot spot) 
new_t2d   = orig_t2d + 0

#ckpt      = torch.load('/net/scratch/jgershon/models/autofold4_seq2str_base.pt', map_location=torch.device('cpu'))
#ckpt       = torch.load('/home/jgershon/projects/c6d_diff/BFF/autofold4/experiments/2209235seqdiffV4_accum4_str5_aa5_continued/models/BFF_4.pt', map_location=torch.device('cpu'))
ckpt = torch.load('/net/scratch/lisanza/diffuse_3track_fullcon/models/BFF_last.pt', map_location=torch.device('cpu'))

weights   = ckpt['model_state_dict']

print("original weights")
print('templ_emb.emb.weight', weights['templ_emb.emb.weight'].shape)
print('templ_emb.emb_t1d.weight', weights['templ_emb.emb_t1d.weight'].shape)

# weights['templ_emb.emb.weight'] # Original shape: (64, 88)
# weights['templ_emb.emb_t1d.weight'] # Original shape: (64, 52)

# Adding 2D embedding features
# d_t1d*2+d_t2d
if True:
    #pt1_add_dim     = new_t2d - orig_t2d
    pt2_add_dim     = new_t1d - orig_t1d
    pt3_add_dim     = new_t1d - orig_t1d 
    
    #pt1_emb_zeros   = torch.zeros(64, pt1_add_dim)
    pt2_emb_zeros   = torch.zeros(64, pt2_add_dim)
    pt3_emb_zeros   = torch.zeros(64, pt3_add_dim)

    '''
    The way that the t2d input to embedding is created is not straightforward
    It looks like this:

        # Prepare 2D template features
        left = t1d.unsqueeze(3).expand(-1,-1,-1,L,-1)
        right = t1d.unsqueeze(2).expand(-1,-1,L,-1,-1)

        templ = torch.cat((t2d, left, right), -1) # (B, T, L, L, 109)
        templ = self.emb(templ) # Template templures (B, T, L, L, d_templ)
    '''

    #new_emb_weights = torch.cat( (weights['templ_emb.emb.weight'][:,:orig_t2d], pt1_emb_zeros), dim=-1 )
    #new_emb_weights = torch.cat( (new_emb_weights, weights['templ_emb.emb.weight'][:,orig_t2d:orig_t2d+orig_t1d], pt2_emb_zeros), dim=-1 )
    new_emb_weights = torch.cat( (weights['templ_emb.emb.weight'][:,:orig_t2d+orig_t1d], pt2_emb_zeros), dim=-1 )
    new_emb_weights = torch.cat( (new_emb_weights, weights['templ_emb.emb.weight'][:,orig_t2d+orig_t1d:], pt3_emb_zeros), dim=-1 )

    #new_emb_weights = torch.cat( (pt1_emb_weights, pt2_emb_weights, pt3_emb_weights), dim=-1 )
    
# Adding 1D embedding features
# d_t1d+d_tor
if True:
    t1d_weights_dim = new_t1d + orig_dtor
    t1d_add_dim     = t1d_weights_dim - weights['templ_emb.emb_t1d.weight'].shape[1] #52
    
    t1d_zeros       = torch.zeros(64, t1d_add_dim)
    new_t1d_weights = torch.cat( (weights['templ_emb.emb_t1d.weight'][:,:orig_t1d], t1d_zeros), dim=-1 )
    new_t1d_weights = torch.cat( (new_t1d_weights, weights['templ_emb.emb_t1d.weight'][:,orig_t1d:]), dim=-1 )

weights['templ_emb.emb.weight']     = new_emb_weights
weights['templ_emb.emb_t1d.weight'] = new_t1d_weights
print("new t1d weights dim")
print(new_t1d_weights.shape)

ckpt['model_state_dict'] = weights
#torch.save(ckpt, './models/t1d_23_t2d_44_BFF_last.pt')
#torch.save(ckpt, './models/t1d_24_t2d_44_BFF_last.pt')
torch.save(ckpt, '/net/scratch/lisanza/projects/diffusion/models/t1d_29_t2d_44_BFF_SE3big_2.pt')
#torch.save(ckpt, './models/t1d_24_t2d_44_BFF_diffV4_accum4_str5_aa5_continued.pt')



def predict_batch_vae(model,x_sc,y):
	m,v,bm,bv,theta,beta,zz =  model(x_sc)
	return theta,beta,zz,y

def predict_batch_flow(model,z1,z2):
    w,d = model(z1,z2)
    return w,d 


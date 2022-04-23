from torchvision.transforms import Resize, Pad

def get_label_encoders():
    to_id = {}
    for i, c in enumerate(["red","blue","black","orange"]):
        for j, v in enumerate([str(i) for i in range(1,14)] + ["j"]):
            to_id[v+'-'+c] = (j+1)*10 + (i+1)

    to_name = {v:k for k,v in to_id.items()}
    return to_id, to_name

def resize(image, targets, img_sz):
    if img_sz is None:
        return image, targets

    x = image
    c, h, w = x.shape
    padding = (0, 0, max(h, w)-w, max(h, w)-h)
    x = Pad(padding)(x)
    image = Resize(img_sz)(x)
    
    ratio = img_sz / max(h, w)
    targets["boxes"] = (targets["boxes"]*ratio).int()
    targets["area"] = (targets["area"]*ratio*ratio).int()
    
    return image, targets
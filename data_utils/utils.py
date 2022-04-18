def get_label_encoders():
    to_id = {}
    for i, c in enumerate(["red","blue","black","orange"]):
        for j, v in enumerate([str(i) for i in range(1,14)] + ["j"]):
            to_id[v+'-'+c] = (j+1)*10 + (i+1)

    to_name = {v:k for k,v in to_id.items()}
    return to_id, to_name
import os
import clip
import torch
from torchvision.datasets import CIFAR100

# Load the model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
model, preprocess = clip.load('ViT-B/32', device)

# Download the dataset
cifar100 = CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=False)

multiple_text_input = True

eval_list = []

def most_common(lst):
    return max(set(lst), key=lst.count)

# Prepare the inputs
for each_image_idx in range(len(cifar100)):
    print("Processing..... image {}".format(each_image_idx))
    image, class_id = cifar100[each_image_idx]

    image_input = preprocess(image).unsqueeze(0).to(device)
    #text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in cifar100.classes]).to(device)
    # More text input
    text_input1 = [clip.tokenize(f"A photo of a {c}") for c in cifar100.classes]
    text_input2 = [clip.tokenize(f"This is a photo of {c}") for c in cifar100.classes]
    text_input3 = [clip.tokenize(f"Wow look at {c}!") for c in cifar100.classes]
    text_input4 = [clip.tokenize(f"What is {c}?") for c in cifar100.classes]
    text_input5 = [clip.tokenize(f"How do you think about {c}?") for c in cifar100.classes]
    if multiple_text_input is True:
        text_inputs1 = torch.cat((text_input1)).to(device)
        text_inputs2 = torch.cat((text_input2)).to(device)
        text_inputs3 = torch.cat((text_input3)).to(device)
        text_inputs4 = torch.cat((text_input4)).to(device)
        text_inputs5 = torch.cat((text_input5)).to(device)
    else:
        text_inputs = torch.cat((text_input1)).to(device)

    #text_inputs = torch.cat((text_input5)).to(device)
    #text_inputs = torch.cat((text_input1 + text_input2 + text_input3 + text_input4 + text_input5)).to(device)
    #import pdb; pdb.set_trace()

    # Calculate features
    if multiple_text_input is True:
        with torch.no_grad():
            image_features = model.encode_image(image_input)
            text_features1 = model.encode_text(text_inputs1)
            text_features2 = model.encode_text(text_inputs2)
            text_features3 = model.encode_text(text_inputs3)
            text_features4 = model.encode_text(text_inputs4)
            text_features5 = model.encode_text(text_inputs5)

        # Pick the top 5 most similar labels for the image
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features1 /= text_features1.norm(dim=-1, keepdim=True)
        text_features2 /= text_features2.norm(dim=-1, keepdim=True)
        text_features3 /= text_features3.norm(dim=-1, keepdim=True)
        text_features4 /= text_features4.norm(dim=-1, keepdim=True)
        text_features5 /= text_features5.norm(dim=-1, keepdim=True)
        similarity1 = (100.0 * image_features @ text_features1.T).softmax(dim=-1)
        similarity2 = (100.0 * image_features @ text_features2.T).softmax(dim=-1)
        similarity3 = (100.0 * image_features @ text_features3.T).softmax(dim=-1)
        similarity4 = (100.0 * image_features @ text_features4.T).softmax(dim=-1)
        similarity5 = (100.0 * image_features @ text_features5.T).softmax(dim=-1)
        values1, indices1 = similarity1[0].topk(5)
        values2, indices2 = similarity2[0].topk(5)
        values3, indices3 = similarity3[0].topk(5)
        values4, indices4 = similarity4[0].topk(5)
        values5, indices5 = similarity5[0].topk(5)
    else:
        with torch.no_grad():
            image_features = model.encode_image(image_input)
            text_features = model.encode_text(text_inputs)

        # Pick the top 5 most similar labels for the image
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        values, indices = similarity[0].topk(5)

    # Print the result
    """
    print("\nTop predictions:\n")
    for value, index in zip(values, indices):
        print(f"{cifar100.classes[index]:>16s}: {100 * value.item():.2f}%")
    """

    # Evaluation
    if multiple_text_input is True:
        pred_index1 = int(indices1[torch.argmax(values1)])
        pred_index2 = int(indices2[torch.argmax(values2)])
        pred_index3 = int(indices3[torch.argmax(values3)])
        pred_index4 = int(indices4[torch.argmax(values4)])
        pred_index5 = int(indices5[torch.argmax(values5)])

        pred_index_list = []
        #pred_index_list.append(pred_index1)
        #pred_index_list.append(pred_index2)
        pred_index_list.append(pred_index3)
        #pred_index_list.append(pred_index4)
        #pred_index_list.append(pred_index5)

        pred_index = most_common(pred_index_list)

        if class_id == pred_index:
            eval_list.append(1)
        else:
            eval_list.append(0)
    else:
        pred_index = int(indices[torch.argmax(values)])
        if class_id == pred_index:
            eval_list.append(1)
        else:
            eval_list.append(0)
        
    print(sum(eval_list)/len(eval_list))
    
# Final evaluation
print("Overall accuracy:\n")
final_acc = sum(eval_list) / len(eval_list)
print(round(final_acc*100, 2))
import pdb; pdb.set_trace()
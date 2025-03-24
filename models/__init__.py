from models.resnet import resnet18, resnet50
from models.vit import vit_base_patch16_224
# resnet18,resnet50对应resnet.py中resnet18,50两个函数，用来创建模型
# 如args.model_names=resnet18则该代码等价于=>
# model = resnet18(num_classes=args.classes_num, pretrained=args.pre_trained)
 # 即调用resnet.py文件中的resnet18方法创建模型
model_dict = {
    'resnet18': resnet18,
    'resnet50': resnet50,
    'vit_base_patch16_224': vit_base_patch16_224,
}

def create_model(model_name, num_classes):   
    return model_dict[model_name](num_classes = num_classes)
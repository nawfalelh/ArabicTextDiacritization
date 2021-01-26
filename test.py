from model import model
from utils import predict_and_diacritize

model.load_weights('checkpoints/epoch35.ckpt')

s = 'ينمو الإسلام ببطء في تايوان ويمثل حوالي 0.3% من السكان. يوجد حوالي 60.000 مسلم في تايوان'

print(predict_and_diacritize(model, s))


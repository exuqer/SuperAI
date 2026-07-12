import sys
sys.path.insert(0, '.')

from superai.service import SuperAIService

s = SuperAIService()
print('Service initialized successfully')
s.close()
print('Service closed successfully')

from gpytorch_GPR import GPR

# Esegui
gpr = GPR('train.csv', 'test.csv')

# Accedi ai risultati
print(gpr.results)
print(gpr.results['Balanced Accuracy'])
print(gpr.results['Y_pred'])

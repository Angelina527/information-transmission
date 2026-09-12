import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# 1. Фиксация сида генератора случайных чисел (для воспроизводимости результатов)
np.random.seed(42)

# 2. Параметры Варианта №9 (Нормальное распределение)
mu = 1.0        # Математическое ожидание (сдвиг)
sigma = 0.2     # Среднеквадратичное отклонение (масштаб)
n_samples = 10000  # Размер выборки (10^4)

# 3. Генерация выборки объемом 10^4
sample = np.random.normal(loc=mu, scale=sigma, size=n_samples)

# 4. Формирование сетки значений X для построения гладких кривых
# Берем интервал в пределах [mu - 4*sigma, mu + 4*sigma]
x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 1000)

# 5. Вычисление аналитической (теоретической) плотности вероятности
pdf_analytical = stats.norm.pdf(x, loc=mu, scale=sigma)

# 6. Непараметрическая оценка: Ядерная оценка плотности (KDE)
kde = stats.gaussian_kde(sample)
pdf_kde = kde(x)

# 7. Построение графиков
plt.figure(figsize=(10, 6))

# Оценка 1: Гистограмма (непараметрический метод)
plt.hist(sample, bins=50, density=True, alpha=0.5, color='skyblue', 
         edgecolor='black', label='Гистограмма (непараметрическая оценка)')

# Оценка 2: Ядерная оценка плотности / KDE (непараметрический метод)
plt.plot(x, pdf_kde, color='green', linestyle='--', linewidth=2, 
         label='Ядерная оценка (KDE)')

# Теоретическая кривая: Аналитическая плотность
plt.plot(x, pdf_analytical, color='red', linewidth=2, 
         label='Аналитическая плотность $f_X(x)$')

# Оформление графика
plt.title('Сравнение аналитической плотности и непараметрических оценок\n'
          'Нормальное распределение ($\mu=1, \sigma=0.2$, $N=10^4$)', fontsize=13)
plt.xlabel('$x$', fontsize=12)
plt.ylabel('Плотность вероятности $f(x)$', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, linestyle=':', alpha=0.7)
plt.tight_layout()

# Отображение графика
plt.show()
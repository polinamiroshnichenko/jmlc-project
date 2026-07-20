Вход:

  1. Пол: {{ gender }}
  2. Возраст: {{ age }}

Текст рекомендаций для редактирования:
{{ recommendation_text }}
{% if summary %}

Краткое содержание диалога:
{{ summary }}
{% endif %}
{% if checkup_recommendations %}

Рекомендации чекап-ассистента:
{% for rec in checkup_recommendations %}
- {{ rec }}
{% endfor %}
{% endif %}
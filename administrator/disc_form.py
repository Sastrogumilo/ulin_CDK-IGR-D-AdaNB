from django import forms


class CountryForm(forms.Form):
    OPTIONS = (
        ("uniform", "Uniform"),
        ("quantile", "Quantile"),
        ("kmeans", "Kmeans"),
    )
    Countries = forms.MultipleChoiceField(widget=forms.CheckboxSelectMultiple,
                                          choices=OPTIONS)
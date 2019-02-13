from django import forms
from twitterclone.models import Post,Survey

class SurveyForm(forms.ModelForm):
        class Meta:
                model = Survey
                fields = ['withdrawdata','age','gender','suggestions']
class PostForm(forms.ModelForm):
	class Meta:
		model = Post
		fields = ['title','text']

		

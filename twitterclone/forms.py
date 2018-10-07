from django import forms
from twitterclone.models import Post


class PostForm(forms.ModelForm):
	class Meta:
		model = Post
		fields = ['title','text','videoname']

		

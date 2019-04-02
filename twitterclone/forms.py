from django import forms
from twitterclone.models import Post,Survey
from twitterclone.utils import validate_uuid4
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User

class UserCreationForm(UserCreationForm):
	referral = forms.CharField(required=False, label='Referral Code (optional)')

	class Meta:
		model = User
		fields = ("username", "password1", "password2", "referral")

	def clean_referral(self):
		referral = self.cleaned_data["referral"].replace('-', '')
		if referral != '' and not validate_uuid4(referral):
			raise forms.ValidationError('Invalid referral code, please verify the referral code is valid')

		return referral

	def save(self, commit=True):
		user = super(UserCreationForm, self).save(commit=False)
		user.referral = self.cleaned_data["referral"]
		if commit:
			user.save()
		return user

class SurveyForm(forms.ModelForm):
		class Meta:
				model = Survey
				fields = ['withdrawdata','age','gender','suggestions']
class PostForm(forms.ModelForm):
	class Meta:
		model = Post
		fields = ['title','text']

		

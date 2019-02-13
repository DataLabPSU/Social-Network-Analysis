from django.shortcuts import render, redirect, render_to_response, get_object_or_404
from django.template import RequestContext
from .models import Post, Comment, Share, Profile, Message
from . import forms
from django.contrib.auth.models import User
from django.http import HttpResponse
from django.contrib.auth import login, authenticate
from django.contrib.auth.forms import UserCreationForm
import datetime
import os


def home(request):
	#if request.user == User.objects.get(username='root'): 
	#	Share.objects.filter().delete()
	'''
	if request.user == User.objects.get(username='admin'):  
		os.chdir('media/videos/')
		for i in os.listdir():
			if i[-3:] == 'mp4':
				user = User.objects.get(username='admin')
				temp = Post.objects.create(author=user)
				temp.title = 'Video'
				temp.text = 'This is a video'
				temp.videoname = i

				# fetch video label
				videoid = i[:-3].split('_')[-1]
				f = open("Fake_Video_Labels.txt", "r")
				videos = f.readlines()
				for video in videos:
					if videoid in video:
						videolabel = video
						break
				
				# add video label
				temp.videolabels = videolabel.split('.')[-1].strip()  

				if (i[:-3]+"en.vtt") in os.listdir():
				   temp.subtitles = i[:-3]+"en.vtt"
				temp.real = 0 
				temp.save()
	   	
		os.chdir('temp/') 
		for i in os.listdir():
			if i[-3:] == 'mp4':
				try:
					user = User.objects.get(username='admin')
					temp = Post.objects.create(author=user)
					temp.title = 'Video'
					temp.text = 'This is a video'
					temp.videoname = i
					
					# fetch video label
					videoid = i[:-3].split('_')[-1]
					f = open("True_Video_Labels.txt", "r")
					videos = f.readlines()
					for video in videos:
						if videoid in video:
							videolabel = video
							break
					
					# add video label
					temp.videolabels = videolabel.split('.')[-1].strip()  

					if (i[:-3]+"en.vtt") in os.listdir():
						temp.subtitles = i[:-3]+"en.vtt"
					temp.real = 1
					temp.save()
				except:
					pass
	'''
	'''
	for i in range(5):
			user = User.objects.get(username='vic')
			temp = Post.objects.create(author=user)
			temp.title = 'Fake'
			temp.text = 'This is a fake post'
			temp.real = 0
			temp.save()
   '''
 # if datetime.datetime.now().hour > 12:
	#    return render(request,'twitterclone/end.html')
	d = {}
	print(request.POST)
	followerscount = {}
	'''
	for user in User.objects.all():
		temp = abs(user.profile.credibilityscore) * (user.profile.real + 1) / (user.profile.real + user.profile.fake + 2)
		temp2 = 1 - abs(user.profile.credibilityscore)
		temp3 = 1 - (user.profile.real + 1) / (user.profile.real + user.profile.fake + 2)
   
		d[user.username] = temp / (temp + temp2 * temp3)
		 
		temp = User.objects.get(username=user.username)
		temp.profile.credibilityscore = d[user.username]
		temp.save()
	'''
	if request.user.is_authenticated:

		for user in User.objects.all():
			temp = abs(user.profile.credibilityscore) * (user.profile.real + 1) / (user.profile.real + user.profile.fake + 2)
			temp2 = 1 - abs(user.profile.credibilityscore)
			temp3 = 1 - (user.profile.real + 1) / (user.profile.real + user.profile.fake + 2)
   
			d[user.username] = temp / (temp + temp2 * temp3)
		 
			temp = User.objects.get(username=user.username)
			temp.profile.credibilityscore = d[user.username]
			temp.save()
		if request.method == 'POST':
			try:
				follow = request.POST['follows']
				user = request.user
				user.profile.following = user.profile.following + " " + follow
				user.save()
			except:
				pass

			try:
				postid = request.POST['share']
				post = Post.objects.get(id=postid)
				if request.user != post.author:
					newshare = Share.objects.create(shared=request.user)
					newshare.postid = postid
					try:
						newshare.comment = request.POST['sharecomment']
					except Exception as e:
						print(str(e))
						pass
					newshare.save()
					user = request.user
					if post.real == 0:
						user.profile.fake = user.profile.fake + 1
					else:
						user.profile.real = user.profile.real + 1
					user.save()
					for user in User.objects.all():
						if request.user.username in user.profile.following.split(" "):
							user.profile.notifications = user.profile.notifications + 1
							user.profile.notificationsString += "Post shared by " + str(request.user) + " at " + str(datetime.datetime.now()) + "|"
							user.save()

			except Exception as e:
				pass

			try:

				user = User.objects.get(pk=request.user.id)
				temp = set(user.profile.following.split(" "))
			 
				temp.remove(request.POST['follow'])
				
				user.profile.following = ' '.join(temp)
				user.save()
			except:
				pass
			try:
				user = User.objects.get(pk=request.user.id)
				post = Post.objects.get(pk=request.POST['unique'])
				if post.author != user and request.POST['unique'] not in user.profile.liked:
					post.likes += 1

					# append video labels to user labels
					user.profile.labels = user.profile.labels + post.videolabels + "|"
					print(user.profile.labels)
					post.updated = datetime.datetime.now()
					post.save()
					if post.real == 0:
						user.profile.fake = user.profile.fake + 1
					else:
						user.profile.real = user.profile.real + 1
					user.profile.liked = user.profile.liked + " " + request.POST['unique']
					user.save()

				elif post.author != user and request.POST['unique'] in user.profile.liked:
					temp = user.profile.liked.split(" ")
					temp.remove(request.POST['unique'])
					user.profile.liked = ' '.join(temp)

					# remove first instance of post videolabels 
					user.profile.labels.replace(post.videolabels + "|", "", 1)
					print(user.profile.labels)
					post.likes -= 1
					if post.real == 0:
						user.profile.fake = user.profile.fake - 1
					else:
						user.profile.real = user.profile.real - 1
					post.save()
					user.save()
			except:
				pass
			try:
				comment = request.POST['placeholder']
				newcomment = Comment.objects.create(author=request.user)
				newcomment.post = Post.objects.get(pk=request.POST['submit'])
				newcomment.text = comment
				temp = Post.objects.get(pk=request.POST['submit'])
				temp.updated = datetime.datetime.now()
				temp.save()
				newcomment.save()
				for user in User.objects.all():
					if request.user.username in user.profile.following.split(" "):
						user.profile.notifications = user.profile.notifications + 1
						user.save()
			except:
				pass
		user = User.objects.get(pk=request.user.id)

		posts = Post.objects.filter(author=request.user)
		postlist = list(posts)
		shares = Share.objects.filter(shared=request.user)
		for z in shares:
			content = Post.objects.get(id=z.postid)
			content.author.username = content.author.username + ' Retweeted by ' + z.shared.username
			content.created_date = z.date
			content.sharecomment = z.comment

			postlist.append(content)
		comments = Comment.objects.all()
		if user.profile.following != "":
			for i in set(user.profile.following.split(" ")):
				if i != "":
				   
					following = User.objects.get(username=i)
					otherposts = Post.objects.filter(author=following)
					posts = posts | otherposts
					postlist.extend(list(otherposts))
					sharedposts = Share.objects.filter(shared=following)
					for z in sharedposts:
						content = Post.objects.get(id=z.postid)
						content.author.username = content.author.username + ' Shared by ' + z.shared.username
						content.created_date = z.date
						content.sharecomment = z.comment
						postlist.append(content)
		notificationsString = request.user.profile.notificationsString.split("|")
		userlist = User.objects.exclude(pk=request.user.id)
		finaloutput = []
		temp = (Profile.objects.all().order_by('-credibilityscore'))[:15]
		dictionary = {}
		print(postlist)
		postlist.sort(key=lambda r:r.updated,reverse=True) 
		print(postlist)
		for i in User.objects.all():
			for z in set(i.profile.following.split(" ")):
				try:
				   dictionary[z] += 1
				except:
				   dictionary[z] = 1
		for i in temp:
			if i.user != request.user and i.user.username not in request.user.profile.following.split(" "):
				 
				 try:
				   finaloutput.append([i.user,dictionary[i.user.username]])
				 except:
				   finaloutput.append([i.user,0])              
		following = []
		for i in set(user.profile.following.split(" ")[1:]):
			following.append(User.objects.get(username=i))
		followeenum = 0
		for i in User.objects.all():
			if request.user.username in i.profile.following.split(" "):
				followeenum += 1
		print(finaloutput)
		for i in finaloutput:
			if i[0] in following:
				finaloutput.remove([i[0],i[1]])
		
		context = {
			'posts': postlist,
			'comments': comments,
			'FOLLOWING': following,
			'test': (user.profile.following.split(" ")),
			'currentuser': request.user,
			'notifications': notificationsString,
			'users': userlist,
			'numfollowers': finaloutput,
			'curfollowersnum': len(following),
			'curfolloweesnum': followeenum,
			'image': request.user.profile.imagename
		}

		return render(request, "twitterclone/home.html", context)
	else:
		# send to consent form if not logged in
		return render(request, "twitterclone/agree.html")
		#return redirect('login')


def signup(request):
	if request.method == 'POST':
		form = UserCreationForm(request.POST)
		if form.is_valid():
			form.save()
			username = form.cleaned_data.get('username')
			raw_password = form.cleaned_data.get('password1')
			user = authenticate(username=username, password=raw_password)
			login(request, user)
			return redirect('login')
	else:
		form = UserCreationForm()
	return render(request, 'twitterclone/signup.html', {'form': form})
def instructions(request):

	return render(request,'twitterclone/instructions.html')
def survey(request):
   
	temp = User.objects.get(pk=request.user.id)
	context = {
	'user': temp,
	}
	if request.method == 'POST':
		form = forms.SurveyForm(request.POST)
		if form.is_valid():
			instance = form.save(commit=False)
			instance.author = request.user
			instance.save()

			return render(request,'twitterclone/final.html',context)
	else: 
		form = forms.SurveyForm()
	return render(request,'twitterclone/survey.html',{'form':form})
def addpost(request):
	if request.method == 'POST':
		form = forms.PostForm(request.POST)
		if form.is_valid():
			instance = form.save(commit=False)
			instance.author = request.user
			instance.save()
			return redirect('home')
	else:
		form = forms.PostForm()

	return render(request, 'twitterclone/create.html', {'form': form})


def notification(request):
	temp = request.user.profile.notificationsString
	num = request.user.profile.notifications
	context = {
		'notifications': temp.split("|"),
		'num': num,
	}
	return render(request, 'twitterclone/notification.html', context)


def testfollow(request):
	if request.method == 'POST':
		user = User.objects.get(pk=request.user.id)
		user.profile.following = user.profile.following + " " + request.POST['follow']
		user.save()

	userlist = User.objects.exclude(pk=request.user.id)
	context = {
		'users': userlist
	}
	return render(request, 'twitterclone/follow.html', context)


def pick(request):
	if request.method == 'POST':
		user = User.objects.get(pk=request.user.id)
		user.profile.imagename = request.POST['hidden']
		user.save()
		return redirect('home')
	os.chdir('media/images/')
	images = [i for i in os.listdir()[:15]]
	images = [i for i in images if i[-3:] != 'jpg']
	context = {
		'images': images,
		'user': request.user,
	}
	print(context)
	return render(request, 'twitterclone/settings.html', context)

def final(request):
	temp = User.objects.get(pk=request.user.id)
	context = {
	'user': temp,
	}
	return render(request, 'twitterclone/final.html',context)
def pm(request):
	if request.method == 'POST':
		temp = Message.objects.create(text=request.POST['message'])
		temp.recipient = request.POST['to']
	d = {}
	for i in Message.objects.all():
		if i.recipient == request.user.username or i.author == request.user:
			if i.recipient == request.user.username:
				try:
					d[i.author.username].append(i.text)
				except:
					d[i.author.username] = [i.text]
			else:
				try:
					d[i.recipient].append(i.text)
				except:
					d[i.recipient] = [i.text]
	d['vic'] = ['hello there', 'hi']
	context = {
		'messages': d
	}
	print(d)
	return render(request, 'twitterclone/pm.html', context)

from django.shortcuts import render, redirect, render_to_response, get_object_or_404
from django.template import RequestContext
from .models import Post, Comment, Share
from . import forms
from django.contrib.auth.models import User
from django.http import HttpResponse
from django.contrib.auth import login, authenticate
from django.contrib.auth.forms import UserCreationForm
import datetime


def home(request):
    if request.user.is_authenticated:
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
                    newshare.save()
                    for user in User.objects.all():
                        if request.user.username in user.profile.following.split(" "):
                            user.profile.notifications = user.profile.notifications + 1
                            user.profile.notificationsString += "Post shared by " + str(request.user) + " at " + str(datetime.datetime.now()) + "|"
                            user.save()

            except Exception as e:
                pass

            try:

                user = User.objects.get(pk=request.user.id)
                temp = user.profile.following.split(" ")
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
                    post.save()
                    user.profile.liked = user.profile.liked + " " + request.POST['unique']
                    user.save()
                elif post.author != user and request.POST['unique'] in user.profile.liked:
                    temp = user.profile.liked.split(" ")
                    temp.remove(request.POST['unique'])
                    user.profile.liked = ' '.join(temp)
                    post.likes -= 1
                    post.save()
                    user.save()
            except:
                pass
            try:
                comment = request.POST['placeholder']
                newcomment = Comment.objects.create(author=request.user)
                newcomment.post = Post.objects.get(pk=request.POST['submit'])
                newcomment.text = comment
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
            postlist.append(content)
            print(postlist)
            print(set(postlist))
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
                        postlist.append(content)
        notificationsString = request.user.profile.notificationsString.split("|")
        userlist = User.objects.exclude(pk=request.user.id)
        finaloutput = []
        for i in set(user.profile.following.split(" ")[1:]):
            tempuser = User.objects.get(username=i)
            numfollowers = len(set(tempuser.profile.following.split(" ")[1:]))
            finaloutput.append([i, numfollowers])
        context = {
            'posts': postlist,
            'comments': comments,
            'FOLLOWING': set(user.profile.following.split(" ")[1:]),
            'test': (user.profile.following.split(" ")),
            'currentuser': request.user,
            'notifications': notificationsString,
            'users': userlist,
            'numfollowers': finaloutput,
        }

        return render(request, "twitterclone/home.html", context)
    else:
        return redirect('login')


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

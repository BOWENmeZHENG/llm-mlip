from flask import (
    Blueprint, flash, g, redirect, render_template, request, url_for
)
from werkzeug.exceptions import abort
from AnnoApp.db import get_db
from AnnoApp.pyutils import split_para, write_anno

bp = Blueprint('blog', __name__)

@bp.route('/')
def index():
    db = get_db()
    posts = db.execute(
        'SELECT p.id, title, body, annotation, created'
        ' FROM post p'
        ' ORDER BY created DESC'
    ).fetchall()
    return render_template('blog/index.html', posts=posts)

@bp.route('/create', methods=('GET', 'POST'))
def create():
    if request.method == 'POST':
        title = request.form['title']
        body = request.form['body']
        annotation = request.form['annotation']
        db = get_db()
        db.execute(
            'INSERT INTO post (title, body, annotation)'
            ' VALUES (?, ?, ?)',
            (title, body, annotation)
        )
        db.commit()
        posts = db.execute(
            'SELECT p.id, title, body, annotation, created'
            ' FROM post p'
            ' ORDER BY created DESC'
        ).fetchall()
        return render_template('blog/index.html', posts=posts)

    return render_template('blog/create.html')

@bp.route('/<int:id>/annotate', methods=('GET', 'POST'))
def annotate(id):
    post = get_post(id)
    if request.method == 'GET':
        text = post['body']
        annotation = post['annotation'] # it's working
        word_list = split_para(text)
        write_anno(f'annotate_{id}', word_list)
        
        return render_template(f'blog/annotate_{id}.html', post=word_list, annotation_exist=annotation, ID=id)
    

def get_post(id, check_author=True):
    post = get_db().execute(
        'SELECT p.id, title, body, annotation, created'
        ' FROM post p'
        ' WHERE p.id = ?',
        (id,)
    ).fetchone()

    return post

@bp.route('/<int:id>/update', methods=('GET', 'POST'))
def update(id):
    post = get_post(id)
    db = get_db()
    if request.method == 'POST':
        title = request.form['title']
        body = request.form['body']
        annotation = request.form['annotation']
        db.execute(
            'UPDATE post SET title = ?, body = ?, annotation = ?'
            ' WHERE id = ?',
            (title, body, annotation, id)
        )
        db.commit()
        posts = db.execute(
            'SELECT p.id, title, body, annotation, created'
            ' FROM post p'
            ' ORDER BY created DESC'
        ).fetchall()
        return render_template('blog/index.html', posts=posts)

    return render_template('blog/update.html', post=post)

@bp.route('/<int:id>/delete', methods=('POST',))
def delete(id):
    db = get_db()
    db.execute('DELETE FROM post WHERE id = ?', (id,))
    db.commit()
    flash(f"Record has been deleted!")
    posts = db.execute(
        'SELECT p.id, title, body, annotation, created'
        ' FROM post p'
        ' ORDER BY created DESC'
    ).fetchall()
    return render_template('blog/index.html', posts=posts)

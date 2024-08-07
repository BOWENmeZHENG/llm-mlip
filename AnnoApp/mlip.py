from flask import (
    Blueprint, flash, g, redirect, render_template, request, url_for
)
from werkzeug.exceptions import abort
from AnnoApp.db import get_db
from AnnoApp.pyutils import split_para, write_anno

bp = Blueprint('mlip', __name__)

@bp.route('/')
def index():
    db = get_db()
    texts = db.execute(
        'SELECT p.id, title, body, annotation, created'
        ' FROM mlip p'
        ' ORDER BY created DESC'
    ).fetchall()
    return render_template('mlip/index.html', texts=texts)

@bp.route('/create', methods=('GET', 'POST'))
def create():
    if request.method == 'POST':
        title = request.form['title']
        body = request.form['body']
        annotation = request.form['annotation']
        db = get_db()
        db.execute(
            'INSERT INTO mlip (title, body, annotation)'
            ' VALUES (?, ?, ?)',
            (title, body, annotation)
        )
        db.commit()
        texts = db.execute(
            'SELECT p.id, title, body, annotation, created'
            ' FROM mlip p'
            ' ORDER BY created DESC'
        ).fetchall()
        return render_template('mlip/index.html', texts=texts)

    return render_template('mlip/create.html')

@bp.route('/<int:id>/annotate', methods=('GET', 'POST'))
def annotate(id):
    text = get_text(id)
    if request.method == 'GET':
        t = text['body']
        annotation = text['annotation'] # it's working
        word_list = split_para(t)
        write_anno(f'annotate_{id}', word_list)
        
        return render_template(f'mlip/annotate_{id}.html', text=word_list, annotation_exist=annotation, ID=id)
    

def get_text(id, check_author=True):
    text = get_db().execute(
        'SELECT p.id, title, body, annotation, created'
        ' FROM mlip p'
        ' WHERE p.id = ?',
        (id,)
    ).fetchone()

    return text

@bp.route('/<int:id>/update', methods=('GET', 'POST'))
def update(id):
    text = get_text(id)
    db = get_db()
    if request.method == 'POST':
        title = request.form['title']
        body = request.form['body']
        annotation = request.form['annotation']
        db.execute(
            'UPDATE mlip SET title = ?, body = ?, annotation = ?'
            ' WHERE id = ?',
            (title, body, annotation, id)
        )
        db.commit()
        texts = db.execute(
            'SELECT p.id, title, body, annotation, created'
            ' FROM mlip p'
            ' ORDER BY created DESC'
        ).fetchall()
        return render_template('mlip/index.html', texts=texts)

    return render_template('mlip/update.html', text=text)

@bp.route('/<int:id>/delete', methods=('POST',))
def delete(id):
    db = get_db()
    db.execute('DELETE FROM mlip WHERE id = ?', (id,))
    db.commit()
    flash(f"Record has been deleted!")
    texts = db.execute(
        'SELECT p.id, title, body, annotation, created'
        ' FROM mlip p'
        ' ORDER BY created DESC'
    ).fetchall()
    return render_template('mlip/index.html', texts=texts)

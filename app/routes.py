from flask import render_template,flash,redirect,url_for,request,send_from_directory,send_file
from app import app
from app.program import computing_s,computing_m,hla_convert,svg_path,check_peptide,check_mhc,wrapper_file_process, binding_score_from_mhcflurry_s


@app.route('/result')
def result():
    peptide = request.args.get('peptide')
    mhc = request.args.get('mhc')
    is_checked = request.args.get('is_checked')  # str
    score,flag= computing_s(peptide,mhc)
    if is_checked == 'True':
        binding = binding_score_from_mhcflurry_s(peptide,mhc)
    else:
        binding = "NA"
    p,m,i,b = computing_m(peptide,mhc,is_checked)
    m3 = hla_convert(mhc)
    path = ["/static/{0}_positive_9.png".format(m3),"/static/{0}_negative_9.png".format(m3),"/static/{0}_positive_10.png".format(m3),"/static/{0}_negative_9.png".format(m3)]
    return render_template('result.html',peptide=peptide,mhc=mhc,score=score,p=p,m=m,i=i,m3=m3,path=path,binding=binding,b=b,flag=flag)


@app.route('/download')
def download():
    return send_file("download/result.txt",as_attachment=True)


@app.route('/learn_more')
def learn_more():
    return render_template('learn_more.html')




@app.route('/',methods=['GET','POST'])
def home():
    if request.method == 'POST':
        peptide = request.form['peptide']
        mhc = request.form['mhc']
        if peptide and mhc:
            if check_peptide(peptide) and check_mhc(mhc):
                is_checked = request.form.get('binding_s') != None  # boolean
                return redirect(url_for('result',peptide=peptide,mhc=mhc,is_checked=str(is_checked)))
            else:
                flash("Please check your peptide and MHC input!")
                flash("Peptide: length of 9 or 10, valid amino acid one letter!")
                flash("MHC: format like HLA-A*0201")
                return render_template('submit.html')
        else:
            uploaded_file = request.files['file']
            uploaded_file.save("./uploaded/multiple_query.txt")
            cond = wrapper_file_process()
            if not cond:
                flash("please check your file format:")
                flash("file should contains two columns,delimiter is comma")
                flash("first column is the peptide, make sure length is 9 or 10, and only contains valid amino acid letter")
                flash("second column is the HLA, make sure the format like HLA-A*0201 ")
                return render_template('submit.html')
            else:
                return redirect(url_for('download'))
    return render_template('submit.html')






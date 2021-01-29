import multer from 'multer';
import fs from 'fs';
import { PythonShell } from 'python-shell';
import Pusher from 'pusher';
import { v4 as uuid } from 'uuid';
const path = require('path');

export const config = {
    api: {
        bodyParser: false,
    },
}

var storage = (dest, filename) => multer.diskStorage({
    destination: function(req, file, cb) {
        cb(null, dest);
    },
    filename: function(req, file, cb) {
        cb(null, filename);
    }
});
var upload = (dest, filename) => multer({ storage: storage(dest, filename) });

const pusher = new Pusher({
    appId: process.env.PUSHER_APP_ID,
    key: process.env.PUSHER_APP_KEY,
    secret: process.env.PUSHER_APP_SECRET,
    cluster: process.env.PUSHER_APP_CLUSTER,
    useTLS: true,
});


export default async function handler(req, res) {
    if (req.method === 'POST') {
        const filename = new Date().getTime()+'toanalyze.mp4';
        upload('model', filename).single("video")(req, {}, err => {
            // console.log('in upload');
            if (!req.file || !req.file.originalname.endsWith('.mp4')) {
                res.status(400).send('No correct mp4 video file sent.');
                return;
            }
            const withVideo = req.body?.withVideo === 'true';
            console.log('Starting script..');
            // generating unique id for communicating update to client
            const clientId = uuid();
            const outputFilename = `output-${clientId}.mp4`;
            const pyprocess = new PythonShell(process.env.ROOT + '/model/analyze.py', {
                args: withVideo ? ['-i '+filename, '-o '+outputFilename] : ['-i '+filename]
            });
            pyprocess.on('message', msg => {
                console.log(msg);
                // if update message -> send to client
                if (msg.startsWith('[PROGRESS%]')) {
                    pusher.trigger(clientId, 'progress', { value: parseInt(msg.substring(12)) });
                }
                // if results message -> send to client
                else if (msg.startsWith('[RESULTS]')) {
                    pusher.trigger(clientId, 'results', JSON.parse(msg.substring(10)));
                }
            });
            pyprocess.end((err, code, signal) => {
                fs.unlinkSync(process.env.ROOT + '/model/' + filename);
                if (err) {
                    console.log(err);
                    pusher.trigger(clientId, 'error', { msg: 'The script terminated with error.' });
                } else {  
                    console.log('The exit code was: ' + code);
                    console.log('The exit signal was: ' + signal);
                    console.log('Done');
                    pusher.trigger(clientId, 'success', { msg: 'Finished.', videopath: withVideo ? outputFilename : undefined });
                }

                // delete generated file after timeout
                setTimeout(() => {
                    try {
                        fs.unlinkSync(process.env.ROOT + '/public/'+outputFilename);
                    } catch (err) {
                        console.log(err);
                    }
                }, parseInt(process.env.DELETE_AFTER) * 60 * 1000); // after 5 min
            });
            res.status(200).send({
                status: 'analyzing..',
                clientId,
            });
          });
    } else {
        res.status(400).send();
    }
}